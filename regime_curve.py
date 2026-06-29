"""regime_curve.py — assemble the C1/C2/C3 leakage-regime sweep into one table+figure.

Reads test_metrics.json from each regime's run dir and reports accuracy/AUROC/
AUPRC/MCC as a function of how much protein-level leakage the split permits:

  C1  both proteins seen in train        (random edge split)   runs/c1
  C2  exactly one protein novel          (novel-protein holdout) runs/c2
  C3  both proteins novel  (Bernett strict, the headline 0.660)  runs/bmse2

Missing runs are skipped with a note, so this is safe to run while jobs are still
training. Writes regime_curve.png + prints a markdown table for the docs.
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C

REGIMES = [
    ("C1", "both seen (random edges)", "runs/c1"),
    ("C2", "one protein novel", "runs/c2"),
    ("C3", "both novel (Bernett strict)", "runs/bmse2"),
]


def load(run: str):
    p = C.PROJECT_ROOT / run / "test_metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))["test"]


def main():
    rows = []
    print("\n| Regime | Construction | acc | AUROC | AUPRC | MCC |")
    print("|---|---|---|---|---|---|")
    for code, desc, run in REGIMES:
        m = load(run)
        if m is None:
            print(f"| {code} | {desc} | _pending_ | | | |")
            continue
        rows.append((code, m))
        print(f"| {code} | {desc} | {m['acc']:.3f} | {m['auroc']:.3f} "
              f"| {m['auprc']:.3f} | {m['mcc']:.3f} |")

    if len(rows) < 2:
        print("\n[regime_curve] <2 regimes available — figure skipped.", flush=True)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    for metric, marker in (("acc", "o"), ("auroc", "s"), ("auprc", "^")):
        ax.plot(codes, [r[1][metric] for r in rows], marker + "-", label=metric)
    ax.axhline(0.65, ls="--", c="grey", lw=1, label="0.65 barrier")
    ax.set_xlabel("leakage regime (more novel proteins ->)")
    ax.set_ylabel("test metric")
    ax.set_title("PPI performance vs. leakage regime (one cache)")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    out = C.PROJECT_ROOT / "regime_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\n[regime_curve] figure -> {out}", flush=True)


if __name__ == "__main__":
    main()
