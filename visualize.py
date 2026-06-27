"""
visualize.py — validation metrics & comparative charts for the BMSE pipeline.

Reads a training run directory (history.json, test_metrics.json, test_preds.npz)
and renders:
  * learning_trajectory.png  — loss + val metrics (Acc/F1/MCC/AUROC/AUPRC) vs epoch.
  * auroc_auprc_curves.png   — ROC and PR curves on the held-out test split, with
                               the 0.65 leakage-free accuracy barrier annotated.
  * embedding_ablation.png   — test metrics across embedding configurations
                               (read from runs/*/test_metrics.json) for the
                               ESM2-only / ProstT5-only / bilingual comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

METRICS = ["acc", "f1", "mcc", "auroc", "auprc"]


def learning_trajectory(run: Path, out: Path):
    hist = json.load(open(run / "history.json"))
    ep = [h["epoch"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ep, [h["train_loss"] for h in hist], "k-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Training loss")
    ax1.grid(alpha=.3); ax1.legend()
    for m in METRICS:
        ax2.plot(ep, [h.get(f"val_{m}") for h in hist], label=f"val {m}")
    ax2.axhline(0.65, ls="--", c="r", alpha=.6, label="0.65 barrier")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.set_title("Validation metrics")
    ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "learning_trajectory.png", dpi=150)
    plt.close(fig)
    print("[viz] learning_trajectory.png", flush=True)


def auroc_auprc_curves(run: Path, out: Path):
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    d = np.load(run / "test_preds.npz"); y, p = d["y"], d["p"]
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    acc = ((p >= .5).astype(int) == y).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y,p):.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC (test)")
    ax1.grid(alpha=.3); ax1.legend()
    ax2.plot(rec, prec, label=f"AUPRC={average_precision_score(y,p):.4f}")
    ax2.axhline(y.mean(), ls=":", c="gray", label=f"prevalence={y.mean():.2f}")
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.set_title("PR (test)")
    ax2.grid(alpha=.3); ax2.legend()
    fig.suptitle(f"Test accuracy = {acc:.4f}  (leakage-free 0.65 barrier)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "auroc_auprc_curves.png", dpi=150)
    plt.close(fig)
    print(f"[viz] auroc_auprc_curves.png (test acc={acc:.4f})", flush=True)


def embedding_ablation(runs_root: Path, out: Path):
    """Bar chart over runs that wrote test_metrics.json (e.g. esm-only, prostt5-only,
    bilingual). Run name -> label via the directory name."""
    runs = sorted(p.parent for p in runs_root.glob("*/test_metrics.json"))
    if not runs:
        print("[viz] no ablation runs found; skipping embedding_ablation", flush=True)
        return
    labels = [r.name for r in runs]
    data = {m: [json.load(open(r / "test_metrics.json"))["test"][m] for r in runs]
            for m in METRICS}
    x = np.arange(len(labels)); w = 0.15
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 5))
    for i, m in enumerate(METRICS):
        ax.bar(x + (i - 2) * w, data[m], w, label=m)
    ax.axhline(0.65, ls="--", c="r", alpha=.6, label="0.65 barrier")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("test score"); ax.set_title("Embedding ablation (test)")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(out / "embedding_ablation.png", dpi=150)
    plt.close(fig)
    print("[viz] embedding_ablation.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(C.RUN_DIR / "bmse"))
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    run = Path(a.run); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    learning_trajectory(run, out)
    auroc_auprc_curves(run, out)
    embedding_ablation(C.RUN_DIR, out)


if __name__ == "__main__":
    main()
