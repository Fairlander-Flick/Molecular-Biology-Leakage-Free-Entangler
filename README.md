# Molecular-Biology — Leakage-Free Entangler

A multimodal, structure-aware **Protein–Protein Interaction (PPI)** prediction pipeline
engineered to break the **0.65 leakage-free accuracy barrier** on the gold-standard
human PPI benchmark of **Bernett et al.** (Figshare DOI
[10.6084/m9.figshare.21591618.v3](https://doi.org/10.6084/m9.figshare.21591618.v3)).

The benchmark is *leakage-free*: train/val/test partitions are split so that no protein
(and no close homolog) is shared across splits, which collapses the inflated accuracies
(~0.8+) reported by sequence-similarity-leaking methods down to a realistic ~0.5–0.65.
Beating 0.65 *honestly* is the objective.

> 📊 **Class presentation script:** [`PRESENTATION.md`](PRESENTATION.md) — slide-by-slide
> talking points + plain-language notes for the C1/C2/C3 leakage-regime findings.

## Core idea — Bilingual Multi-Scale Entangler (BMSE)

Two protein "languages" are fused:

| Modality | Encoder | Signal |
|----------|---------|--------|
| Evolutionary / sequence | `facebook/esm2_t33_650M_UR50D` | per-residue contextual embeddings |
| Structure-aware | `Rostlab/ProstT5` | 3Di structural-token-aware embeddings |

These frozen per-residue embeddings feed:
- **Cross-Chain Attention** — bidirectional residue↔residue interface mapping between the two chains.
- **Multi-Scale CNN Branch** — physical motifs across receptive fields from length 16 → 512.

## Hardware-aware execution (NHR@FAU TinyGPU)

All compute runs through SLURM on the FAU TinyGPU cluster. See [`HARDWARE.md`](HARDWARE.md)
for the live hardware scan. Approved strategy:

- **Embedding extraction** → parallel SLURM **array across idle RTX 3080 nodes** (8 GPU/node).
- **BMSE training** → **A100** when available, with **RTX 3080 fallback** (the head trains over
  *cached* embeddings, so it is not VRAM-bound).
- **Precision** → `bfloat16` autocast (Ampere-native, no `GradScaler`) + **TF32** matmul.
- **Compilation** → `torch.compile` on the BMSE model (selective); PLM extraction stays eager.

Central knobs live in [`config.py`](config.py).

## Status & headline result

The **BMSE baseline clears the 0.65 barrier**, leakage-clean:

| Model | Test Acc | AUROC | AUPRC | MCC | leakage (degree_corr) |
|-------|---------:|------:|------:|----:|----------------------:|
| **BMSE (ESM2+ProstT5)** | **0.660** | **0.722** | 0.708 | 0.320 | −0.01 (clean) |

## Progress so far — what we did → what we got

| # | Step | What we did | What we got |
|---|------|-------------|-------------|
| 1 | Hardware/repo | Profiled TinyGPU; chose RTX 3080 array + bf16/TF32 | a100/v100 inaccessible → all on RTX 3080 |
| 2 | Data + embeddings | Bernett v3; ESM2+ProstT5 per-residue, two-tier 1024; 32-shard array | `embeddings.h5` — 11,018 proteins, 5.82 M residues, 26.9 GB |
| 3 | Model | BMSE: bilingual fusion + multi-scale CNN + cross-chain attention | 4.85 M params |
| 4 | Training | bf16 AMP, AdamW, contrastive segment-shuffle, degree-debias | first (buggy) test acc 0.642 |
| 5 | **Bug found+fixed** | Nondeterministic set-ordering halved shard coverage (7,429/11,018) → tie-break by id; re-extract + retrain | cache 11,018/11,018 unique; **test acc 0.642 → 0.660** |
| 6 | Headline | Clean BMSE on full benchmark | **acc 0.660 / AUROC 0.722 / MCC 0.320, leakage-clean** |
| 7 | Track A (co-evolution) | mmseqs Swiss-Prot search; phylo-profile features | phylo standalone AUROC **0.537** (weak); coevo MI computing |
| 8 | Track B (structure) | ESMFold infeasible on 10 GB → ESM2 contact maps | structure-only ≈ **random (0.518)** |
| 9 | Fusion | LightGBM stack BMSE + phylo + structure | ≈ BMSE alone (orthogonal tracks add ~0) |

**Conclusion:** on Bernett-strict (C3) the orthogonal signals don't help — ~0.66–0.68
is the honest ceiling for sequence methods. Full detail in [`RESULTS.md`](RESULTS.md).

## Next objective → ~0.75 (see [`ROADMAP.md`](ROADMAP.md))

0.75 is a property of the split's difficulty, not the model. Next we move to a
legitimate, less-adversarial benchmark (a dataset with overlapping proteins →
the **C2 regime**, "predict new partners of a known protein"), run this same
pipeline, tune + ensemble, and **report per split regime** so every number carries
its context. Resume commands in [`RUNBOOK.md`](RUNBOOK.md).

## Pipeline stages

1. **Hardware profiling & repo init** — `gpu_profile.py`, `config.py`. ✅
2. **Dataset & embedding extraction** — `dataset.py` → ragged-flat HDF5 cache. ✅
3. **Architecture** — `models.py` (BMSE: Cross-Chain Attention + Multi-Scale CNN). ✅
4. **Training** — `train.py` (bf16 AMP, AdamW, contrastive segment-shuffle, degree-debias check). ✅
5. **Validation & visualization** — `visualize.py` (Loss/Acc/F1/MCC/AUROC/AUPRC + curves). ✅

**Barrier-breaking tracks (toward 0.71):**
- **Track A — co-evolution / phylogenetic profiling** (`msa/`, `coevo/`): mmseqs vs
  UniRef50/Swiss-Prot → `phylo_features.py`, `coevo_features.py`.
- **Track B — structure** (`struct/`): ESMFold infeasible on 10 GB → ESM2 contact-map
  descriptors (`contacts.py`).
- **Fusion** (`fusion/`): LightGBM stacking BMSE + phylo + structure + coevolution.

## Environment

```bash
# conda env `kaggle`: torch 2.5.1+cu121, transformers 5.12.0
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
```
