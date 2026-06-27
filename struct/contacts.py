"""
contacts.py — ESM2 attention-based contact-map structural descriptors (Track B).

ESMFold (3B trunk + structure module) does not fit the 10GB RTX 3080s, and the
larger-VRAM partitions are inaccessible. ESM2's attention-derived contact map is
a hardware-feasible structural signal: from the predicted L x L contact map we
derive topology descriptors (contact order, range distribution, globularity)
without 3D folding. Covers proteins up to 1024 residues (vs ESMFold's <=512).

Writes the same desc_shard_*.npz schema struct_features.py consumes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import dataset as D

OUT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(C.HF_HOME))
DESC = ["L", "n_contacts_norm", "contact_density", "rel_contact_order",
        "short_range_frac", "med_range_frac", "long_range_frac",
        "mean_prob", "max_offdiag_prob", "globularity"]


def descriptors(c: np.ndarray) -> list[float]:
    L = c.shape[0]
    iu = np.triu_indices(L, k=3)
    probs = c[iu]
    contact = probs > 0.5
    seps = (iu[1] - iu[0])[contact]
    n = int(contact.sum())
    if n:
        rco = float(seps.mean() / L)
        sr = float((seps <= 5).mean()); mr = float(((seps > 5) & (seps < 12)).mean())
        lr = float((seps >= 12).mean())
    else:
        rco = sr = mr = lr = 0.0
    # globularity: fraction of residues with >=1 long-range contact
    lrc = (c > 0.5) & (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 12)
    glob = float((lrc.any(1)).mean())
    return [L, n / L, float(contact.mean()), rco, sr, mr, lr,
            float(probs.mean()), float(probs.max() if probs.size else 0.0), glob]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmModel
    _, id2seq, ordered = D.build_manifest()
    ids = ordered[a.shard::a.num_shards]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(C.ESM2_MODEL)
    model = EsmModel.from_pretrained(C.ESM2_MODEL, attn_implementation="eager",
                                     torch_dtype=torch.float16).to(device).eval()

    rows, kept = [], []
    for pid in ids:
        seq = id2seq[pid][: a.max_len]
        enc = tok([seq], return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                c = model.predict_contacts(enc["input_ids"], enc["attention_mask"])[0]
            rows.append(descriptors(c.float().cpu().numpy())); kept.append(pid)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    out = OUT / f"desc_shard_{a.shard:03d}_{a.num_shards:03d}.npz"
    np.savez(out, X=np.array(rows, np.float32),
             ids=np.array(kept, dtype="S15"), feat_names=np.array(DESC))
    print(f"[contacts] shard {a.shard}: {len(kept)}/{len(ids)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
