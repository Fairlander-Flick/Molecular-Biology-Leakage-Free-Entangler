# ROADMAP — what we're doing next

## Where we are
Our BMSE model clears the 0.65 leakage-free barrier on the Bernett-strict (C3)
benchmark: **test acc 0.660 / AUROC 0.722, leakage-clean** (degree_corr −0.01).
We tested orthogonal pairwise signals (phylogenetic profiling, structure) and
they don't move the number on this split — that's settled.

## Decision: characterise performance across the C1/C2/C3 leakage regimes — on our own cache
We've confirmed 0.75 is **not** honestly attainable on Bernett-strict (C3) — that
ceiling is ~0.68 for any sequence method, and anything higher there is leakage.
PPI accuracy is governed by the **split regime** (how much the test proteins
overlap training). Rather than chase a bigger number on a foreign dataset, we
report the **full regime curve** so each accuracy carries its setting. This is the
Bernett-paper framing, but fully controlled: same 11,018 proteins, same 26.9 GB
embedding cache, just re-partitioned. **No new download, no new GPU extraction.**

### The realised plan (no HIPPIE needed)
Originally we assumed Bernett's protein-disjoint partitions had "no C2 pairs to
make." Not so: pooling all 274,327 edges and marking a **random 15% of proteins
novel** yields genuine **one-novel (C2)** edges — within-group edges where exactly
one endpoint is held out. So all three regimes come from the one cache
(`resplit.py`):

1. **C1** — random edge split → test 99.6% both-seen. Proof-of-mechanism (~0.80).
2. **C2** — novel-protein holdout → test edges have exactly one novel endpoint, the
   realistic "predict new partners of a known protein" task (~0.74 honest target).
3. **C3** — the original Bernett strict split = **0.660** (already have it, `runs/bmse2`).

`dataset.py` reads these via `PPI_SPLIT_DIR={c1,c2}`; `regime_curve.py` assembles
the table + figure. Then: **tune + ensemble** the realistic C2 point (best epoch was
#2 — we're undertuned: lower LR, larger `d_model`, 5-seed ensemble, ~+0.02–0.03).
HIPPIE/HuRI stays available later as *independent external replication* if wanted.

### Why this is honest, not number-inflation
C1 is explicitly the **homology-leaking baseline** — it shows exactly where inflated
PPI numbers come from. We never quote it as "the result"; it anchors the upper end
of a curve whose realistic point is C2 and whose adversarial point is C3 (0.660).

### Guardrails (so the number stays real)
- Always report the split regime (C1/C2/C3) next to the accuracy.
- No graph-topology features (degree, common neighbors), no GO terms derived from
  interactions, no STRING labels as features — those inflate dishonestly.
- Keep the degree-leakage check on every run.

See `RESULTS.md` for the full log of what we did and got; `RUNBOOK.md` for exact
commands to resume.
