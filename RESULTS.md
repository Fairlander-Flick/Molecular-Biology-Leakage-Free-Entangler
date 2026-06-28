# RESULTS & PROJECT LOG

Chronological record of everything done on this pipeline, with results. Goal:
break the **0.65 leakage-free accuracy barrier** (stretch: 0.71) on the Bernett
gold-standard human PPI benchmark (Figshare 10.6084/m9.figshare.21591618 v3).
Cluster: NHR@FAU TinyGPU (SLURM, RTX 3080 10 GB; a100/v100 inaccessible).

## Headline result

| Model | Test Acc | AUROC | AUPRC | MCC | degree_corr (leakage) |
|-------|---------:|------:|------:|----:|----------------------:|
| **BMSE (ESM2+ProstT5), full clean cache** | **0.660** | **0.722** | 0.708 | 0.320 | −0.01 (clean) |
| BMSE — partial cache (pre-bugfix) | 0.642 | 0.710 | 0.697 | 0.290 | −0.05 |
| Fusion (BMSE+phylo+struct) | 0.646\* | 0.709\* | 0.696\* | 0.293\* | — |

\* measured on the pre-bugfix subset; orthogonal tracks added ~nothing (see below).

**The BMSE baseline clears the 0.65 barrier (0.660 acc / 0.722 AUROC) and is
leakage-clean** (predictions uncorrelated with protein node degree, the dominant
non-biological shortcut). 0.71 accuracy remains the stretch goal.

## Pipeline (Steps 1–5, baseline)

1. **Hardware/repo init** — `config.py`, `gpu_profile.py`, `HARDWARE.md`. Strategy:
   RTX 3080 extraction array; a100 training (fell back to rtx3080, a100 capped);
   bf16 + TF32; selective torch.compile.
2. **Data + embeddings** — `dataset.py`: md5-verified Figshare v3 download; manifest
   (train=Intra1 163,019 / val=Intra0 59,260 / test=Intra2 52,048 pairs; 11,018
   unique proteins; 173 pairs dropped for missing FASTA). Two-tier length policy
   (full ≤1024; head512+tail512 beyond). ESM2-650M (1280-d) + ProstT5 `<AA2fold>`
   (1024-d) per-residue, bf16, length-bucketed; ragged-flat HDF5
   (`embeddings.h5`, 11,018 proteins, 5.82 M residues, 26.9 GB) via 32-shard
   RTX 3080 array + streaming merge.
3. **Architecture** — `models.py`: **BMSE** (4.85 M params): bilingual per-residue
   fusion → multi-scale dilated CNN (kernels {3,9,17,33}) → bidirectional
   Cross-Chain Attention (8 heads ×32, 2 layers) → symmetric interaction head
   `[a;b;|a−b|;a·b]` + contrastive projection.
4. **Training** — `train.py`: bf16, AdamW, warmup→cosine, early-stop on val AUPRC;
   contrastive segment-shuffle (C3PI-style NT-Xent, λ=0.2); degree-debias check;
   `--ablation` flag.
5. **Visualization** — `visualize.py`: learning trajectory, ROC/PR curves,
   embedding ablation.

## Critical bug found & fixed

`build_manifest` sorted a Python `set` by length only; hash-randomized tie order
made `ordered[shard::N]` **inconsistent across SLURM array tasks** → 3,589 proteins
never embedded, others duplicated (11,018 rows but only 7,429 unique) → **~half of
all pairs silently dropped from training/eval.** Fixed by tie-breaking on protein
id (deterministic in every process). Re-extracted → verified 11,018/11,018 unique
→ retrained: test acc **0.642 → 0.660**. (Commit `fix(critical): deterministic
protein ordering`.)

## Ablations (val, pre-bugfix; rerun on clean cache pending)

- **ESM2-only ≈ full bilingual** (val acc ~0.64, AUROC ~0.70).
- **ProstT5-only weaker** (val acc ~0.62, AUROC ~0.67).
→ ESM2 carries most single-sequence signal; ProstT5 structure-tokens add little.

## Barrier-breaking effort (A + B research tracks)

Single-sequence embeddings saturate near the leakage-free ceiling; gains past it
need **pairwise** signal. Two tracks were built and tested:

### Track A — co-evolution / phylogenetic profiling (`msa/`, `coevo/`)
- mmseqs2 search of 11,018 queries vs seqTaxDB → `hits.tsv` (taxonomy) + a3m MSAs.
- **Phylogenetic profiling** (`coevo/phylo_features.py`): TF-IDF-weighted taxon
  co-occurrence similarity per pair. **Standalone AUROC 0.537** (all feature
  corrs |r|<0.05) — weak; adds ~nothing in fusion.
- **Coevolution** (`coevo/coevo_features.py`): paired-MSA APC-corrected inter-MI
  (numba). In progress on Swiss-Prot MSAs.
- **Swiss-Prot** DB (fast) done; **UniRef50** (deep) downloaded but the search
  **OOM'd in prefilter** (47 GB RAM cap for 1 GPU) — needs `--split-memory-limit`.

### Track B — structure (`struct/`)
- **ESMFold infeasible on 10 GB** (3B trunk OOMs even short proteins; a100/v100
  inaccessible) → pivoted to **ESM2 attention contact-map descriptors**
  (`struct/contacts.py`): contact order, range distribution, globularity.
- **Structure-only fusion ≈ random (acc 0.518)** — per-protein structure doesn't
  encode whether two *specific* proteins dock. Low value; parked in the stack.

### Fusion (`fusion/train_fusion.py`)
LightGBM stacking BMSE feat+prob + phylo + struct (+coevo) with a BMSE+fusion
ensemble. So far **fusion ≈ BMSE alone** — `bmse_prob` dominates feature gain; no
phylo/struct feature reaches the top 20.

## Honest assessment

The orthogonal tracks do **not** rescue the embedding model on this leakage-free
split — consistent with Bernett's pessimism (removing cross-split homology also
flattens co-occurrence signal). The **clean BMSE at 0.660 acc / 0.722 AUROC,
leakage-verified, is a strong honest result that clears 0.65.** Reaching 0.71
accuracy is near-SOTA and remains a genuine research bet riding on deep
co-evolution (UniRef50 + DCA), which the evidence so far does not strongly support.

## Operational gotchas (TinyGPU)
- Login node Ivy Bridge (SSE4 only) → `mmseqs-sse41` for DB ops; AVX2 `mmseqs` on
  compute. Compute nodes offline → prefetch HF models. ESMFold/ProstT5 ship only
  `.bin`; transformers 5.12 blocks torch.load on torch<2.6 → pre-convert to
  safetensors. `sbatch` forbids `--mem` and requires `--gres=gpu` (≤8 CPU/GPU);
  RAM auto-scales to #GPU (~47 GB). Node-local `/scratch` invisible from login.
  h5py handles not fork-safe → open lazily per worker. ESM2 `predict_contacts`
  needs `attn_implementation="eager"`.

See `RUNBOOK.md` for exact resume steps; memory `ppi-entangler-runbook` for handoff.
