#!/bin/bash
#SBATCH --job-name=ppi-train
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#
# BMSE training on a100 (preferred). For the rtx3080 fallback, submit with:
#   sbatch -p rtx3080 --gres=gpu:rtx3080:1 train.sh
# The a100 (510GB RAM) comfortably holds the full per-residue cache in RAM
# (--preload) for max throughput.

set -euo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd "$REPO"
mkdir -p logs runs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Note: torch.compile is left off by default — variable-length padded batches
# trigger frequent recompilation. Add --compile explicitly if using static shapes.
srun "$PY" train.py --preload "$@"
