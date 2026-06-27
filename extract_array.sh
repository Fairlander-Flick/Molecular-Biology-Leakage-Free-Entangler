#!/bin/bash
#SBATCH --job-name=ppi-extract
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/extract_%A_%a.out
#
# Parallel ESM2 + ProstT5 embedding extraction across idle RTX 3080 nodes.
# Each array task embeds a length-balanced shard of the 11,018 unique proteins
# (round-robin over the length-sorted list) and writes one ragged-flat HDF5 shard.
# Merge afterwards:  python dataset.py merge

set -euo pipefail
NUM_SHARDS=32
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python

export HF_HOME=/home/woody/dsaa/dsaa115h/hf_cache
export HF_HUB_OFFLINE=1          # compute nodes have no internet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 10GB card: reduce fragmentation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$REPO"
mkdir -p logs
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" dataset.py extract --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}"
