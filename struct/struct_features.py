"""
struct_features.py — merge ESMFold descriptor shards into per-pair features.

Per-protein descriptors (struct/fold.py) -> symmetric pair features:
  [d_a + d_b, |d_a - d_b|, d_a * d_b]  (order-invariant) + both_folded flag.
Missing structures (long/failed proteins) are median-imputed; an availability
flag lets the fusion model down-weight imputed rows.

Outputs: struct/struct_{train,val,test}.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as D

OUT = Path(__file__).resolve().parent


def load_descriptors():
    desc, names = {}, None
    for f in sorted(OUT.glob("desc_shard_*.npz")):
        d = np.load(f, allow_pickle=True)
        names = list(d["feat_names"])
        for pid, row in zip(d["ids"], d["X"]):
            desc[pid.decode()] = row
    return desc, names


def main():
    pairs_by_split, _, _ = D.build_manifest()
    desc, names = load_descriptors()
    if not desc:
        raise RuntimeError("no desc_shard_*.npz — run fold_array.sh first")
    alld = np.stack(list(desc.values()))
    med = np.median(alld, 0)
    F = alld.shape[1]
    feat_names = ([f"sum_{n}" for n in names] + [f"absdiff_{n}" for n in names]
                  + [f"prod_{n}" for n in names] + ["both_folded"])
    print(f"[struct] {len(desc)} folded proteins, {F} descriptors", flush=True)

    for split, pairs in pairs_by_split.items():
        X, y, ids = [], [], []
        for a, b, lab in pairs:
            da, db = desc.get(a), desc.get(b)
            both = float(da is not None and db is not None)
            da = da if da is not None else med
            db = db if db is not None else med
            X.append(np.concatenate([da + db, np.abs(da - db), da * db, [both]]))
            y.append(lab); ids.append((a, b))
        X = np.array(X, np.float32)
        np.savez(OUT / f"struct_{split}.npz", X=X, y=np.array(y, np.int8),
                 ids=np.array(ids, dtype="S15"), feat_names=np.array(feat_names))
        print(f"[struct] {split}: X={X.shape} both_folded={X[:,-1].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
