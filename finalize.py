"""finalize.py — the master results table: 650M vs 3B single vs 3B ensemble vs
3B anti-overfit, across C1/C2/C3. Single-model metrics come from test_metrics.json;
ensembles are the mean of test_preds.npz probabilities over seeds (same fixed test
split -> aligned rows). Safe to run anytime; missing runs show as '—'.
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, matthews_corrcoef, roc_auc_score)

R = "/home/hpc/dsaa/dsaa115h/ppi-entangler"


def single(run):
    f = os.path.join(R, run, "test_metrics.json")
    return json.load(open(f))["test"] if os.path.exists(f) else None


def ensemble(runs):
    ps, y = [], None
    for run in runs:
        f = os.path.join(R, run, "test_preds.npz")
        if not os.path.exists(f):
            return None
        z = np.load(f)
        ps.append(z["p"]); y = z["y"] if y is None else y
    if not ps:
        return None
    p = np.mean(ps, axis=0); pred = (p > 0.5).astype(int)
    return {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred),
            "mcc": matthews_corrcoef(y, pred), "auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p), "n": len(ps)}


def row(label, m):
    if not m:
        return f"| {label} | _pending_ | | | |"
    return f"| {label} | {m['acc']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['mcc']:.3f} |"


def main():
    print("\n### Master results — accuracy / AUROC / AUPRC / MCC\n")
    print("| Setup | acc | AUROC | AUPRC | MCC |\n|---|---|---|---|---|")
    # C1
    print(row("C1 650M", single("runs/c1")))
    print(row("C1 3B", single("runs/c1_3b")))
    # C2 (the honest realistic regime)
    print(row("C2 650M", single("runs/c2")))
    print(row("C2 3B", single("runs/c2_3b")))
    print(row("C2 3B ensemble(3)", ensemble(["runs/c2_3b", "runs/c2_3b_s2", "runs/c2_3b_s3"])))
    # C3 (the strict leakage-free regime — the Bernett challenge)
    print(row("C3 650M", single("runs/bmse2")))
    print(row("C3 3B", single("runs/c3_3b")))
    print(row("C3 3B ensemble(3)", ensemble(["runs/c3_3b", "runs/c3_3b_s2", "runs/c3_3b_s3"])))
    print(row("C3 3B anti-overfit", single("runs/c3_3b_reg")))
    print("\n(650M C3 baseline = 0.660/0.721; goal = push C3 AUROC up = the scaling rebuttal.)")


if __name__ == "__main__":
    main()
