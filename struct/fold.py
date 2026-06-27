"""
fold.py — ESMFold structural descriptors per protein (Track B).

Folds a shard of proteins with ESMFold and reduces each to a compact, robust
descriptor vector (no need to keep full structures):
  * disorder      : mean/median pLDDT, frac pLDDT<50 / >70 / >90 (disorder is a
                    genuine PPI-relevant signal — IDRs behave differently).
  * global shape  : radius of gyration, end-to-end / Rg, gyration-tensor
                    asphericity (CA only).
  * topology      : relative contact order (mean seq-separation of CA contacts<8A).

Run as a SLURM array over idle RTX 3080s. Long proteins (> --max-len) are skipped
and imputed downstream. Merge -> struct/desc.npz.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import dataset as D

OUT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(C.HF_HOME))
DESC = ["L", "mean_plddt", "median_plddt", "frac_lt50", "frac_gt70", "frac_gt90",
        "rg", "ete_over_rg", "asphericity", "rel_contact_order"]


def descriptors(ca: np.ndarray, plddt: np.ndarray) -> list[float]:
    L = len(ca)
    cen = ca - ca.mean(0)
    rg = float(np.sqrt((cen ** 2).sum(1).mean()))
    ete = float(np.linalg.norm(ca[0] - ca[-1]))
    # gyration tensor asphericity
    T = (cen.T @ cen) / L
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    asph = float((ev[0] - 0.5 * (ev[1] + ev[2])) / (ev.sum() + 1e-9))
    # relative contact order
    d = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    iu = np.triu_indices(L, k=3)
    contacts = d[iu] < 8.0
    seps = (iu[1] - iu[0])[contacts]
    rco = float(seps.mean() / L) if seps.size else 0.0
    return [L, float(plddt.mean()), float(np.median(plddt)),
            float((plddt < 50).mean()), float((plddt > 70).mean()),
            float((plddt > 90).mean()), rg, ete / (rg + 1e-9), asph, rco]


def parse_pdb_ca(pdb: str):
    ca, plddt = [], []
    for line in pdb.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            plddt.append(float(line[60:66]))   # B-factor = pLDDT
    return np.array(ca), np.array(plddt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--max-len", type=int, default=512)   # 10GB VRAM cap
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    _, id2seq, ordered = D.build_manifest()
    ids = ordered[a.shard::a.num_shards]

    tok = AutoTokenizer.from_pretrained(C.ESMFOLD_MODEL)
    model = EsmForProteinFolding.from_pretrained(C.ESMFOLD_MODEL, use_safetensors=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()           # fp16 language trunk to save VRAM
    model.set_chunk_size(64)               # chunked attention -> lower peak memory

    rows, kept = [], []
    for pid in ids:
        seq = id2seq[pid]
        if len(seq) > a.max_len:
            continue
        try:
            with torch.no_grad():
                pdb = model.infer_pdb(seq)
            ca, plddt = parse_pdb_ca(pdb)
            if len(ca) >= 5:
                rows.append(descriptors(ca, plddt)); kept.append(pid)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    out = OUT / f"desc_shard_{a.shard:03d}_{a.num_shards:03d}.npz"
    np.savez(out, X=np.array(rows, np.float32),
             ids=np.array(kept, dtype="S15"), feat_names=np.array(DESC))
    print(f"[fold] shard {a.shard}: folded {len(kept)}/{len(ids)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
