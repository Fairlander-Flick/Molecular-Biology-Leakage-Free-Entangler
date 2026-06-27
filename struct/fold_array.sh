#!/bin/bash
#SBATCH --job-name=ppi-fold
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/fold_%A_%a.out
#
# ESMFold structural descriptors across idle RTX 3080s (Track B).
# Each task folds a length-balanced shard (proteins <=512 res for 10GB VRAM).
# Merge: python struct/struct_features.py

set -euo pipefail
NUM_SHARDS=32
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
export HF_HOME=/home/woody/dsaa/dsaa115h/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"; mkdir -p logs
srun "$PY" struct/fold.py --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" --max-len 512
