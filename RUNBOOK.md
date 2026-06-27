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

## What's done
- **Baseline (Steps 1–5):** `config/dataset/models/train/visualize.py` committed.
  Embedding cache **built**: `woody/ppi-entangler/embeddings/embeddings.h5`
  (11,018 proteins, 5.82M residues). Training running → `runs/{bmse,abl_esm,abl_prostt5}`.
- **Track B (structure):** ESMFold OOMs 10GB → pivoted to ESM2 contact maps
  (`struct/contacts.py`). `struct/struct_{train,val,test}.npz` **built**.
  Finding: structure-only fusion ≈ random (0.518) → low value, kept in stack.
- **Track A (coevolution):** `msa/run_search.sh` (param by `DB`/`TAG`),
  `coevo/phylo_features.py`, `coevo/coevo_features.py` (env `PPI_MSA_DIR`).
  DBs downloading: Swiss-Prot (fast) + UniRef50 (deep) under `woody/dbs`.
- **Fusion:** `predict_bmse.py` + `fusion/train_fusion.py` (LightGBM stack + ensemble).

## Do next (in order)
1. **Swiss-Prot ready** (`woody/dbs/swissprot.dbtype`):
   `DB=woody/dbs/swissprot TAG=sp sbatch msa/run_search.sh`
2. Features: `PPI_MSA_DIR=woody/ppi-entangler/msa/sp $PY coevo/phylo_features.py`
   and `... $PY coevo/coevo_features.py` → `coevo/{phylo,coevo}_{split}.npz`.
3. **UniRef50 ready** (`woody/dbs/uniref50.dbtype`): same with `TAG=uniref50`,
   rerun features with `PPI_MSA_DIR=.../msa/uniref50` (overwrites with deeper feats).
4. **Training done** (`runs/bmse/best.pt`): `$PY predict_bmse.py` → `fusion/bmse_*.npz`;
   `$PY visualize.py` → figures (commit them).
5. **Barrier test:** `$PY fusion/train_fusion.py` → test acc/AUROC/AUPRC/MCC + ensemble.

Full chronological detail: memory `ppi-entangler-runbook`.
