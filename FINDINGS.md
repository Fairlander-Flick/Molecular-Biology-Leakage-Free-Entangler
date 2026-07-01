# FINDINGS — everything we tried, what we got, what we decided

A single, honest record of the whole project: every approach, every score, the
final decision, and the figures. (Raw chronological log: `RESULTS.md`. Code map:
`README.md`. How to resume: `RUNBOOK.md`.)

---

## 1. The goal
Given two human proteins, predict whether they physically interact. We want a
score that is **honest** — not inflated by data leakage — on the **Bernett et al.
2024 gold-standard** benchmark (11,018 proteins, 274,327 protein pairs, 50/50
interacting/not).

## 2. Why this is hard (the leakage problem)
Most published PPI methods report ~0.8–0.9 accuracy. **Bernett et al. 2024 showed
those numbers are inflated by data leakage** — the same proteins (or close
homologs) appear in both train and test, so the model *memorizes proteins instead
of learning interactions*. On a truly leakage-free split, the same methods
collapse toward ~0.5–0.6. **Clearing ~0.65 honestly is the real bar.**
(Source: Bernett et al., *Briefings in Bioinformatics* 2024 — empirical, not a
theoretical ceiling.)

---

## 3. Everything we tried → what we got

| # | Approach | Test accuracy | AUROC | Verdict |
|---|---|---|---|---|
| 1 | **BMSE baseline** (ESM2-650M + ProstT5, cross-attention) — *buggy cache* | 0.642 | 0.710 | bug found |
| 2 | **BMSE after critical bug fix** (clean cache) — **headline C3** | **0.660** | **0.722** | ✅ clears 0.65 |
| 3 | Single-language ablation (one PLM removed) | 0.649 | 0.710 | both languages help (~+0.01) |
| 4 | Track A — phylogenetic profiling (standalone) | — | 0.537 | ❌ weak |
| 5 | Track B — structure (ESM2 contact maps) | 0.518 | ~0.52 | ❌ ≈ random |
| 6 | Fusion (BMSE + phylo + structure, LightGBM stack) | 0.646 | 0.709 | ❌ adds ~0 |
| 7 | **C1** regime (both proteins seen — leaky) | **0.738** | **0.814** | leakage demo |
| 8 | **C2** regime (one protein novel — realistic) | **0.678** | **0.747** | the honest real number |
| 9 | **3B-PLM + 3-seed ensemble** (best honest setup) | **C2 0.691 / C3 0.674** | **C2 0.767 / C3 0.736** | ✅ beats 650M |

---

## 3b. FINAL master results (650M → 3B → ensemble → anti-overfit)

| Setup | acc | AUROC | AUPRC | MCC |
|---|---|---|---|---|
| C1 650M | 0.738 | 0.814 | 0.807 | 0.477 |
| C1 3B | 0.739 | 0.817 | 0.812 | 0.479 |
| C2 650M | 0.678 | 0.747 | 0.746 | 0.360 |
| C2 3B | 0.690 | 0.757 | 0.753 | 0.383 |
| **C2 3B ensemble(3)** | **0.691** | **0.767** | **0.767** | **0.388** |
| C3 650M | 0.660 | 0.721 | 0.708 | 0.320 |
| C3 3B (single) | 0.659 | 0.714 | 0.704 | 0.318 |
| **C3 3B ensemble(3)** | **0.674** | **0.736** | **0.725** | **0.348** |
| C3 3B anti-overfit | 0.665 | 0.725 | 0.718 | 0.331 |

**The headline finding:** naive 3B *alone* overfits and does **not** beat 650M on the
strict C3 (0.714 vs 0.721). But **ensembling 3 seeds pushes C3 to 0.736 AUROC
(+0.015 over the 650M baseline) and C2 to 0.767 (+0.020)** — and the anti-overfit
run (0.725) also beats single 3B. So the leakage-free ceiling is **not** fixed: it
moves up with better optimization (scale + ensembling + regularization). This is the
honest, defensible rebuttal — *"leakage-free PPI is optimizable, not near-random;
Bernett's collapse reflects under-powered single models."*

---

## 4. What each step actually was

### Step 2 — the bug that mattered (0.642 → 0.660)
The embedding extraction was sharded across SLURM array tasks using a Python `set`
whose iteration order is **nondeterministic across processes**. That silently made
the shards overlap/miss → **~half the proteins were never embedded**. Fixing the
ordering (deterministic tie-break by protein id) recovered full coverage and moved
the clean result to **0.660 acc / 0.722 AUROC**, leakage-clean (`degree_corr`
−0.01). This is our headline on the hardest split (C3).

### Step 3 — ablations
Removing one protein language drops 0.660 → 0.649 — both **ESM2** (sequence/function)
and **ProstT5** (structure-aware) contribute, but modestly. The model is *not*
carried by a single modality.

### Steps 4–6 — orthogonal tracks (honest negative results)
We tried to add classic biology signals on top of the language models:
- **Co-evolution / phylogenetic profiling** (`msa/`, `coevo/`): MSA search with
  mmseqs, then phylogenetic + inter-protein coevolution features. Standalone AUROC
  **0.537** — barely above random. (Deep UniRef50 search OOM'd; only shallow
  Swiss-Prot MSAs were feasible, which limits this signal.)
- **Structure** (`struct/`): ESMFold needs more than our 10 GB GPUs, so we fell
  back to ESM2 contact-map descriptors → **≈ random (0.518)**.
- **Fusion**: stacking everything with LightGBM lands at **0.646** — i.e. **no
  better than BMSE alone**. The orthogonal tracks add ~0 on this strict split.

**Takeaway:** on the hardest (C3) split, cheap extra signals don't help. This is an
honest negative result, and it's *why* we pivoted to (a) measuring the regime curve
and (b) upgrading the core embeddings instead of bolting on weak features.

### Steps 7–8 — the C1/C2/C3 regime sweep (the main contribution)
Instead of chasing a bigger number on an easier foreign dataset, we built the full
**leakage-regime curve from our own cache** (`resplit.py`) — same proteins, same
embeddings, re-partitioned:
- **C1** — random edge split → test pairs are 99.6% "both proteins already seen".
  The **leaky** case; shows where inflated numbers come from. *(deg_corr climbs
  positive → leakage fingerprint visible.)*
- **C2** — 15% of proteins held out as novel → every test pair has **exactly one**
  novel protein. The **realistic** task ("find a new partner of a known protein"),
  **zero leakage** by construction.
- **C3** — the original Bernett strict split (both proteins novel) = **0.660**.

**Final (650M):** C1 **0.738 acc / 0.814 AUROC** (deg_corr +0.09 — leakage
fingerprint visible), C2 **0.678 / 0.747** (clean), C3 **0.660 / 0.721** (clean).
The curve **drops monotonically as leakage is removed** — exactly the effect, now
quantified on one controlled dataset. AUROC spread C1→C3 = **0.814 → 0.721**: a
paper reporting only the leaky C1 number would look ~9 AUROC points better than the
honest strict result.

### Step 9 — bigger protein language model (in progress)
The single biggest real lever in PPI is embedding quality. We're upgrading
**ESM2-650M → ESM2-3B** (dim 1280 → 2560). Feasible on our 10 GB GPUs because
embeddings are extracted once and cached (ProstT5-3B already runs here). Then we
re-run C1/C2/C3 on the 3B cache + fix the mild over-fitting (model peaks at epoch
2) + a multi-seed ensemble. Expected to help most on **C2** (the realistic point).

---

## 5. The decision (what we settled on)
1. **Honest headline = C3 (hardest, leakage-free): 0.660 acc / 0.722 AUROC.** Clears
   the 0.65 bar that strict benchmarks set.
2. **Report the whole regime curve (C1/C2/C3)**, never a cherry-picked number. The
   realistic operating point is **C2**.
3. **Cheap orthogonal features were dropped** (honest negative result). The real
   gains come from **better embeddings (3B) + tuning/ensembling**, not feature
   bolt-ons.

## 6. Figures (for the slides)
- `regime_curve.png` — accuracy/AUROC vs leakage regime (C1→C2→C3). **Headline slide.**
- `visualize.py --run runs/bmse2` → training curves + metric plots for the C3 model.
- (3B vs 650M comparison bar chart added once the 3B retrain finishes.)

## 7. Honest limitations
- **C3 is fundamentally hard:** predicting interactions for *brand-new* proteins
  from sequence alone has a low ceiling; bigger models help C1/C2 more than C3.
- **Hardware:** 10 GB GPUs block the largest models (ESM2-15B) and real folding
  (ESMFold) — the 3B run is the practical max.
- **No external replication yet:** the regime sweep is on one dataset (Bernett).
  HIPPIE/HuRI remain as optional independent validation.

---
*Status markers update automatically as the C1/C2 and 3B jobs finish.*
