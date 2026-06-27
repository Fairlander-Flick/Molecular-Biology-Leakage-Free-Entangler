#!/bin/bash
#SBATCH --job-name=ppi-contacts
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/contacts_%A_%a.out
#
# ESM2 contact-map structural descriptors across idle RTX 3080s (Track B).
# ESMFold doesn't fit 10GB; ESM2 contacts do and cover up to 1024 residues.
# Merge: python struct/struct_features.py

set -euo pipefail
NUM_SHARDS=16
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
export HF_HOME=/home/woody/dsaa/dsaa115h/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"; mkdir -p logs
srun "$PY" struct/contacts.py --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" --max-len 1024
