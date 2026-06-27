"""
train_fusion.py — combine BMSE + phylogenetic + structural + coevolution signals.

The single-modality BMSE saturates near the leakage-free ceiling; the gains toward
0.71 must come from the orthogonal pairwise tracks. This stacks all available
feature sources and fits a LightGBM classifier (handles mixed dense/tabular +
missing), then also reports a BMSE+fusion probability ensemble.

Aligns every source by pair id. Sources are optional — runs with whatever exists:
  fusion/bmse_{split}.npz   (prob, feat)         deep model
  coevo/phylo_{split}.npz   (X)                  phylogenetic profiling
  struct/struct_{split}.npz (X)                  ESMFold descriptors
  coevo/coevo_{split}.npz   (X)                  paired-MSA coevolution
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import compute_metrics

SOURCES = {
    "bmse":   ROOT / "fusion" / "bmse_{s}.npz",
    "phylo":  ROOT / "coevo" / "phylo_{s}.npz",
    "struct": ROOT / "struct" / "struct_{s}.npz",
    "coevo":  ROOT / "coevo" / "coevo_{s}.npz",
}


def _key(ids):
    return [tuple(x.decode() if isinstance(x, bytes) else x for x in row) for row in ids]


def load_split(split):
    """Return (feature_matrix, y, names, bmse_prob_or_None) aligned by pair id."""
    blocks, names, order, y_ref, bmse_prob = {}, [], None, None, None
    for src, tmpl in SOURCES.items():
        f = Path(str(tmpl).format(s=split))
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        keys = _key(d["ids"])
        idx = {k: i for i, k in enumerate(keys)}
        if order is None:
            order, y_ref = keys, d["y"]
        rows = np.array([idx[k] for k in order])     # align to first source order
        if src == "bmse":
            bmse_prob = d["prob"][rows]
            X = np.concatenate([d["feat"][rows], bmse_prob[:, None]], 1)
            names += [f"bmse_h{i}" for i in range(d["feat"].shape[1])] + ["bmse_prob"]
        else:
            X = d["X"][rows]
            nm = list(d["feat_names"]) if "feat_names" in d else \
                [f"{src}{i}" for i in range(X.shape[1])]
            names += [f"{src}_{n}" for n in nm]
        blocks[src] = X
    if not blocks:
        raise RuntimeError(f"no feature sources for split={split}")
    X = np.concatenate([blocks[s] for s in blocks], 1)
    print(f"[fusion] {split}: sources={list(blocks)} X={X.shape}", flush=True)
    return X.astype(np.float32), y_ref.astype(int), names, bmse_prob


def main():
    import lightgbm as lgb
    Xtr, ytr, names, _ = load_split("train")
    Xva, yva, _, _ = load_split("val")
    Xte, yte, _, pbm_te = load_split("test")

    dtr = lgb.Dataset(Xtr, ytr, feature_name=names)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.03, num_leaves=63, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=100,
                  max_depth=-1, verbose=-1, seed=42)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

    p_te = model.predict(Xte, num_iteration=model.best_iteration)
    out = Path(__file__).resolve().parent
    print("\n=== FUSION (LightGBM) test ===", flush=True)
    mf = compute_metrics(yte, p_te)
    for k, v in mf.items():
        print(f"  {k}: {v:.4f}", flush=True)
    print(f"  >>> fusion accuracy {mf['acc']:.4f} vs 0.65 / 0.71 <<<", flush=True)

    if pbm_te is not None:
        for w in (0.3, 0.5, 0.7):
            pe = w * p_te + (1 - w) * pbm_te
            me = compute_metrics(yte, pe)
            print(f"[ensemble w_fusion={w}] acc={me['acc']:.4f} auroc={me['auroc']:.4f} "
                  f"auprc={me['auprc']:.4f} mcc={me['mcc']:.4f}", flush=True)

    imp = sorted(zip(names, model.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])[:20]
    print("\n[fusion] top features by gain:", flush=True)
    for n, g in imp:
        print(f"  {n:24s} {g:.0f}", flush=True)
    np.savez(out / "fusion_test_preds.npz", p_fusion=p_te, p_bmse=pbm_te, y=yte)
    model.save_model(str(out / "fusion_lgb.txt"))
    import json
    json.dump({"fusion": mf, "best_iter": model.best_iteration},
              open(out / "fusion_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
