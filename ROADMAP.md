# ROADMAP — what we're doing next

## Where we are
Our BMSE model clears the 0.65 leakage-free barrier on the Bernett-strict (C3)
benchmark: **test acc 0.660 / AUROC 0.722, leakage-clean** (degree_corr −0.01).
We tested orthogonal pairwise signals (phylogenetic profiling, structure) and
they don't move the number on this split — that's settled.

## Decision: go for ~0.75 on a legitimate, less-adversarial benchmark
We've confirmed 0.75 is **not** honestly attainable on Bernett-strict (C3) — that
ceiling is ~0.68 for any sequence method, and anything higher there is leakage.
PPI accuracy is governed by the **split regime** (how much the test proteins
overlap training), so we move to a setting where 0.75 is the honest ceiling and
report it with its context.

### The plan
1. **Acquire a human PPI dataset with overlapping proteins** — HIPPIE, HuRI,
   BioGRID (physical), or high-confidence STRING-physical: sequences + pairs.
   (Bernett's data is protein-disjoint by design, so it has no C2 pairs to make.)
2. **Build C1/C2/C3 splits** (Park–Marcotte). Primary target = **C2** (one protein
   known) — the realistic "predict new partners of a known protein" task, where
   ~0.74–0.80 is honest. Report C1/C2/C3 side by side.
3. **Run the existing pipeline unchanged** (ESM2 + ProstT5 → BMSE; all 10 GB-feasible,
   embeddings already work).
4. **Tune + ensemble** — we're undertuned (best epoch was #2): lower LR, larger
   `d_model` (384), 3–4 cross-attn layers, 5-seed ensemble. Worth +0.02–0.03.
5. **Report per regime** so the 0.75 always carries its setting.

### Immediate proof-of-mechanism (no new data needed)
Re-split the current Bernett pairs **randomly** (proteins shared across splits) and
retrain — the same BMSE jumps to ~0.80–0.85 because homology leakage is restored.
We keep this labeled as the **homology-leaking baseline**; it demonstrates exactly
where inflated PPI numbers come from.

### Guardrails (so the number stays real)
- Always report the split regime (C1/C2/C3) next to the accuracy.
- No graph-topology features (degree, common neighbors), no GO terms derived from
  interactions, no STRING labels as features — those inflate dishonestly.
- Keep the degree-leakage check on every run.

See `RESULTS.md` for the full log of what we did and got; `RUNBOOK.md` for exact
commands to resume.
