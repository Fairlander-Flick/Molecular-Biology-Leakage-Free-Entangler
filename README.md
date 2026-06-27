# Molecular-Biology — Leakage-Free Entangler

A multimodal, structure-aware **Protein–Protein Interaction (PPI)** prediction pipeline
engineered to break the **0.65 leakage-free accuracy barrier** on the gold-standard
human PPI benchmark of **Bernett et al.** (Figshare DOI
[10.6084/m9.figshare.21591618.v3](https://doi.org/10.6084/m9.figshare.21591618.v3)).

The benchmark is *leakage-free*: train/val/test partitions are split so that no protein
(and no close homolog) is shared across splits, which collapses the inflated accuracies
(~0.8+) reported by sequence-similarity-leaking methods down to a realistic ~0.5–0.65.
Beating 0.65 *honestly* is the objective.

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

## Pipeline stages

1. **Hardware profiling & repo init** — `gpu_profile.py`, `config.py` (this commit).
2. **Dataset & embedding extraction** — `dataset.py` → chunked HDF5 cache.
3. **Architecture** — `models.py` (BMSE: Cross-Chain Attention + Multi-Scale CNN).
4. **Training** — `train.py` (AMP, AdamW, contrastive segment-shuffle, taxonomy debiasing).
5. **Validation & visualization** — `visualize.py` (Loss/Acc/F1/MCC/AUROC/AUPRC + curves).

## Environment

```bash
# conda env `kaggle`: torch 2.5.1+cu121, transformers 5.12.0
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
```
