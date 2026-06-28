# RUNBOOK — resume the grind toward 0.71

Pipeline state and exact steps to continue. Goal: beat **0.71 leakage-free** on the
Bernett benchmark. Baseline PLM-embedding model saturates ~0.65; the push comes
from pairwise signal (Track A coevolution/phylogenetic profiling; Track B
structure underdelivers — see below).

## Env (every job)
```bash
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
export HF_HOME=/home/woody/dsaa/dsaa115h/hf_cache HF_HUB_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# rtx3080 partition only (a100/v100 inaccessible). No --mem on GPU jobs.
```

## NEXT OBJECTIVE → ~0.75 (see ROADMAP.md)
Move to a less-adversarial-but-legitimate benchmark (dataset with overlapping
proteins → C2 regime) + tune/ensemble; report per split regime. Bernett-strict (C3)
honest ceiling is ~0.68 — 0.75 there would be leakage.

## Headline result (clean cache)
**BMSE test: acc 0.660 / AUROC 0.722 / AUPRC 0.708 / MCC 0.320, degree_corr −0.01
(leakage-clean) → clears 0.65.** Best ckpt `runs/bmse2/best.pt`. See `RESULTS.md`.

## What's done
- **Baseline (Steps 1–5):** committed. Embedding cache **built & VERIFIED clean**
  (`embeddings.h5`, 11,018/11,018 unique, 5.82M residues, 26.9GB). Clean retrain
  done → `runs/bmse2` (acc 0.660). **Critical sharding bug fixed** (see RESULTS.md):
  nondeterministic set-order halved coverage; tie-break by id in build_manifest.
- **Track B (structure):** ESMFold infeasible on 10GB → ESM2 contact maps
  (`struct/contacts.py`). struct features built. **Finding: ≈ random (0.518).**
- **Track A (coevolution):** Swiss-Prot search DONE (`msa/sp/{hits.tsv,a3m}`).
  Phylo features built — **standalone AUROC 0.537 (weak)**. Coevo (paired-MSA MI)
  running on SP. **UniRef50 search OOM'd in prefilter** (needs `--split-memory-limit`).
- **Fusion:** built; **BMSE+phylo+struct ≈ BMSE alone** (orthogonal tracks add ~0).

## Do next (in order)
1. **Re-run ablations on the clean cache** for the paper:
   `sbatch ... train.sh --ablation esm --out runs/abl_esm2` and `--ablation prostt5 --out runs/abl_prostt5_2`.
2. **Finish coevo SP** (`coevo/coevo_*.npz`), then `$PY predict_bmse.py --ckpt runs/bmse2/best.pt`
   → `fusion/bmse_*.npz`; `$PY fusion/train_fusion.py` → does coevo help?
3. **Deep UniRef50 (path 2, if pursued):** re-run search with
   `mmseqs ... --split-memory-limit 40G` (prefilter OOM'd at 47GB). Then rerun
   phylo/coevo with `PPI_MSA_DIR=.../msa/uniref50` (overwrites with deeper feats).
4. **Figures:** `$PY visualize.py --run runs/bmse2` → commit figures.
5. **Honest call:** baseline 0.660 clears 0.65; 0.71 is the stretch and the
   orthogonal evidence is weak — see RESULTS.md "Honest assessment".

Full chronological detail: memory `ppi-entangler-runbook`.
