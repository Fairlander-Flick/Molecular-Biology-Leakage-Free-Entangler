"""
dataset.py — Bernett gold-standard PPI data + frozen PLM embedding extraction.

Pipeline:
  1. download()      : fetch + md5-verify the Figshare v3 files.
  2. build_manifest(): parse FASTA + Intra{0,1,2} pos/neg pairs -> split manifest,
                       unique-protein list (deduplicated; the heavy work is per
                       *protein*, not per *pair*).
  3. extract()       : embed a shard of unique proteins with ESM2 + ProstT5 and
                       write a ragged-flat HDF5 shard. Run as a SLURM array (one
                       shard per task) across the idle RTX 3080 nodes.
  4. merge()         : concatenate shard files into the final embeddings.h5
                       (ragged-flat [SUM_L, D] per modality + id->offset index +
                       mean/max pooled sidecar).

Two-tier length policy (cap 1024): proteins <=1024 residues are embedded in full;
longer proteins use head-512 + tail-512 (termini carry signal peptides / binding
domains) — reproducible and label-free.

Storage: fp16. ESM2 (1280-d, layer 33) and ProstT5 (<AA2fold> encoder, 1024-d) are
kept separate so the BMSE model fuses them itself.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

import config as C

# Compute nodes are offline: point transformers at the shared pre-staged cache.
os.environ.setdefault("HF_HOME", str(C.HF_HOME))

# --------------------------------------------------------------------------- #
# Figshare v3 manifest (file-id + md5 pinned for reproducibility)
# --------------------------------------------------------------------------- #
FIGSHARE = {
    "Intra0_neg_rr.txt": (41270466, "4d09773d"),
    "Intra0_pos_rr.txt": (41270469, "234b5ce2"),
    "Intra1_neg_rr.txt": (41270472, "fac611b2"),
    "Intra1_pos_rr.txt": (41270475, "f2465329"),
    "Intra2_neg_rr.txt": (41270478, "21868ab1"),
    "Intra2_pos_rr.txt": (41270481, "d7171f30"),
    "human_swissprot_oneliner.fasta": (42862132, "36615eda"),
}
FASTA = "human_swissprot_oneliner.fasta"
# split partition -> role
SPLITS = {"train": "Intra1", "val": "Intra0", "test": "Intra2"}


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def _md5_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def download(data_dir: Path = C.DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (fid, md5) in FIGSHARE.items():
        dst = data_dir / name
        if dst.exists() and _md5_prefix(dst) == md5:
            continue
        url = f"https://ndownloader.figshare.com/files/{fid}"
        print(f"[download] {name} <- {url}", flush=True)
        with urlopen(url) as r, open(dst, "wb") as fh:
            fh.write(r.read())
        got = _md5_prefix(dst)
        if got != md5:
            raise RuntimeError(f"md5 mismatch for {name}: {got} != {md5}")
    print("[download] all files present & verified", flush=True)


# --------------------------------------------------------------------------- #
# 2. Manifest
# --------------------------------------------------------------------------- #
def read_fasta(data_dir: Path = C.DATA_DIR) -> dict[str, str]:
    seqs: dict[str, str] = {}
    pid = None
    with open(data_dir / FASTA) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                pid = line[1:].split()[0]
            elif pid is not None:
                seqs[pid] = line
    return seqs


def read_pairs(partition: str, data_dir: Path = C.DATA_DIR) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for label, pn in ((1, "pos"), (0, "neg")):
        with open(data_dir / f"{partition}_{pn}_rr.txt") as fh:
            for line in fh:
                a, b = line.split()
                rows.append((a, b, label))
    return rows


def build_manifest(data_dir: Path = C.DATA_DIR):
    """Return (pairs_by_split, id2seq_present, ordered_unique_ids).

    Drops pairs that reference a protein with no FASTA sequence.
    `ordered_unique_ids` is sorted by sequence length (ascending) so a SLURM
    array splits the work into length-balanced shards for efficient batching.
    """
    seqs = read_fasta(data_dir)
    pairs_by_split: dict[str, list] = {}
    used: set[str] = set()
    dropped = 0
    for role, part in SPLITS.items():
        kept = []
        for a, b, y in read_pairs(part, data_dir):
            if a in seqs and b in seqs:
                kept.append((a, b, y))
                used.add(a)
                used.add(b)
            else:
                dropped += 1
        pairs_by_split[role] = kept
    ordered = sorted(used, key=lambda p: len(seqs[p]))
    print(f"[manifest] {len(ordered)} unique proteins | dropped {dropped} pairs "
          f"(missing seq) | " + " ".join(f"{r}={len(v)}" for r, v in pairs_by_split.items()),
          flush=True)
    return pairs_by_split, {p: seqs[p] for p in used}, ordered


# --------------------------------------------------------------------------- #
# Two-tier length policy
# --------------------------------------------------------------------------- #
def prep_sequence(seq: str, cap: int = C.MAX_SEQ_LEN_EMB) -> str:
    seq = re.sub(r"[UZOB]", "X", seq.upper())
    if len(seq) > cap:
        half = cap // 2
        seq = seq[:half] + seq[-half:]
    return seq


# --------------------------------------------------------------------------- #
# 3. Embedders (loaded sequentially to fit 10 GB)
# --------------------------------------------------------------------------- #
def _length_batches(seqs: list[str], max_tokens: int):
    """Yield index batches so sum(len) per batch <= max_tokens (>=1 item)."""
    batch, tok = [], 0
    for i, s in enumerate(seqs):
        L = len(s)
        if batch and tok + L > max_tokens:
            yield batch
            batch, tok = [], 0
        batch.append(i)
        tok += L
    if batch:
        yield batch


def embed_esm2(seqs: list[str], device, max_tokens: int):
    import torch
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(C.ESM2_MODEL)
    model = EsmModel.from_pretrained(
        C.ESM2_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    out: list[np.ndarray] = [None] * len(seqs)
    with torch.no_grad():
        for idx in _length_batches(seqs, max_tokens):
            batch = [seqs[i] for i in idx]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hs = model(**enc).last_hidden_state  # [B, L+2, 1280]
            mask = enc["attention_mask"]
            for j, i in enumerate(idx):
                L = int(mask[j].sum()) - 2  # strip CLS/EOS
                out[i] = hs[j, 1:1 + L].float().cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return out


def embed_prostt5(seqs: list[str], device, max_tokens: int):
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(C.PROSTT5_MODEL, do_lower_case=False)
    # ProstT5 ships only pytorch_model.bin; we pre-converted to safetensors to avoid
    # the transformers torch<2.6 torch.load guard. Force the safetensors path.
    model = T5EncoderModel.from_pretrained(
        C.PROSTT5_MODEL, use_safetensors=True,
        torch_dtype=torch.bfloat16).to(device).eval()   # 3B in fp32 = 11GB > 10GB card
    spaced = ["<AA2fold> " + " ".join(list(s)) for s in seqs]
    out: list[np.ndarray] = [None] * len(seqs)
    with torch.no_grad():
        for idx in _length_batches(seqs, max_tokens):
            batch = [spaced[i] for i in idx]
            enc = tok(batch, add_special_tokens=True,
                      padding="longest", return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hs = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"]).last_hidden_state
            mask = enc["attention_mask"]
            for j, i in enumerate(idx):
                L = int(mask[j].sum()) - 2  # strip <AA2fold> prefix + </s>
                out[i] = hs[j, 1:1 + L].float().cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return out


def _pool(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    return np.concatenate([a.mean(0), a.max(0)]).astype(np.float16)


# --------------------------------------------------------------------------- #
# 3b. Extract one shard
# --------------------------------------------------------------------------- #
def extract(shard: int, num_shards: int,
            emb_dir: Path = C.EMB_DIR, data_dir: Path = C.DATA_DIR) -> None:
    import h5py
    import torch
    _, id2seq, ordered = build_manifest(data_dir)
    # round-robin over the length-sorted list -> every shard sees mixed lengths
    ids = ordered[shard::num_shards]
    raw = [id2seq[i] for i in ids]
    seqs = [prep_sequence(s) for s in raw]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] shard {shard}/{num_shards}: {len(ids)} proteins on {device} "
          f"({os.uname().nodename})", flush=True)

    t0 = time.time()
    esm = embed_esm2(seqs, device, C.EXTRACT_MAX_TOKENS["esm2"])
    print(f"[extract] ESM2 done in {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    pro = embed_prostt5(seqs, device, C.EXTRACT_MAX_TOKENS["prostt5"])
    print(f"[extract] ProstT5 done in {time.time()-t1:.0f}s", flush=True)

    emb_dir.mkdir(parents=True, exist_ok=True)
    out = emb_dir / f"shard_{shard:03d}_{num_shards:03d}.h5"
    esm_flat = np.concatenate(esm, 0)
    pro_flat = np.concatenate(pro, 0)
    lengths = np.array([e.shape[0] for e in esm], dtype=np.int32)
    pooled = np.stack([np.concatenate([_pool(e), _pool(p)]) for e, p in zip(esm, pro)])
    with h5py.File(out, "w") as f:
        f.create_dataset("esm2", data=esm_flat, compression="lzf")
        f.create_dataset("prostt5", data=pro_flat, compression="lzf")
        f.create_dataset("lengths", data=lengths)
        f.create_dataset("pooled", data=pooled, compression="lzf")
        f.create_dataset("ids", data=np.array(ids, dtype="S15"))
    print(f"[extract] wrote {out}  ({out.stat().st_size/1e9:.2f} GB)", flush=True)


# --------------------------------------------------------------------------- #
# 4. Merge shards -> final ragged-flat cache
# --------------------------------------------------------------------------- #
def merge(emb_dir: Path = C.EMB_DIR) -> None:
    import h5py
    shards = sorted(emb_dir.glob("shard_*.h5"))
    if not shards:
        raise RuntimeError(f"no shards in {emb_dir}")
    print(f"[merge] {len(shards)} shards", flush=True)
    ids, lengths, pooled, esm_parts, pro_parts = [], [], [], [], []
    for s in shards:
        with h5py.File(s, "r") as f:
            ids.extend(x.decode() for x in f["ids"][:])
            lengths.append(f["lengths"][:])
            pooled.append(f["pooled"][:])
            esm_parts.append(f["esm2"][:])
            pro_parts.append(f["prostt5"][:])
    lengths = np.concatenate(lengths)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    out = emb_dir / "embeddings.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("esm2", data=np.concatenate(esm_parts, 0), compression="lzf")
        f.create_dataset("prostt5", data=np.concatenate(pro_parts, 0), compression="lzf")
        f.create_dataset("pooled", data=np.concatenate(pooled, 0), compression="lzf")
        f.create_dataset("lengths", data=lengths)
        f.create_dataset("offsets", data=offsets)
        f.create_dataset("ids", data=np.array(ids, dtype="S15"))
        f.attrs["esm2_dim"] = C.ESM2_DIM
        f.attrs["prostt5_dim"] = C.PROSTT5_DIM
        f.attrs["cap"] = C.MAX_SEQ_LEN_EMB
    print(f"[merge] wrote {out}: {len(ids)} proteins, "
          f"{offsets[-1]} residues ({out.stat().st_size/1e9:.2f} GB)", flush=True)


# --------------------------------------------------------------------------- #
# 5. Training Dataset (loads cached embeddings)
# --------------------------------------------------------------------------- #
class PPIPairDataset:
    """torch Dataset over cached embeddings. Serves per-residue pairs (+pooled).

    Per-residue tensors are read from HDF5 via offset index; pooled vectors are
    held in RAM. Set `preload=True` to pull the whole per-residue cache into RAM
    (~34 GB) on the big TinyGPU nodes for max throughput.
    """

    def __init__(self, split: str, emb_path: Path = None, data_dir: Path = C.DATA_DIR,
                 max_len: int = C.MAX_SEQ_LEN, preload: bool = False):
        import h5py
        import torch
        self.torch = torch
        emb_path = emb_path or (C.EMB_DIR / "embeddings.h5")
        pairs_by_split, _, _ = build_manifest(data_dir)
        self.f = h5py.File(emb_path, "r")
        self.ids = [x.decode() for x in self.f["ids"][:]]
        self.id2row = {p: i for i, p in enumerate(self.ids)}
        self.offsets = self.f["offsets"][:]
        self.pooled = self.f["pooled"][:]
        self.max_len = max_len
        self.preload = preload
        if preload:
            self.esm2 = self.f["esm2"][:]
            self.prostt5 = self.f["prostt5"][:]
        self.pairs = [(a, b, y) for a, b, y in pairs_by_split[split]
                      if a in self.id2row and b in self.id2row]

    def _residues(self, row: int):
        o0, o1 = int(self.offsets[row]), int(self.offsets[row + 1])
        if self.max_len and (o1 - o0) > self.max_len:
            o1 = o0 + self.max_len
        if self.preload:
            return self.esm2[o0:o1], self.prostt5[o0:o1]
        return self.f["esm2"][o0:o1], self.f["prostt5"][o0:o1]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b, y = self.pairs[i]
        ra, rb = self.id2row[a], self.id2row[b]
        ea, pa = self._residues(ra)
        eb, pb = self._residues(rb)
        t = self.torch
        return {
            "esm_a": t.from_numpy(np.ascontiguousarray(ea)),
            "pro_a": t.from_numpy(np.ascontiguousarray(pa)),
            "esm_b": t.from_numpy(np.ascontiguousarray(eb)),
            "pro_b": t.from_numpy(np.ascontiguousarray(pb)),
            "pool_a": t.from_numpy(self.pooled[ra]),
            "pool_b": t.from_numpy(self.pooled[rb]),
            "y": t.tensor(y, dtype=t.float32),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "manifest", "extract", "merge"])
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    a = ap.parse_args()
    if a.cmd == "download":
        download()
    elif a.cmd == "manifest":
        build_manifest()
    elif a.cmd == "extract":
        download()
        extract(a.shard, a.num_shards)
    elif a.cmd == "merge":
        merge()


if __name__ == "__main__":
    main()
