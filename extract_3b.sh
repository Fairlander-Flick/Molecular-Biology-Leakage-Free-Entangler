#!/bin/bash
#SBATCH --job-name=ppi-ext3b
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-63
#SBATCH --output=logs/ext3b_%A_%a.out
#
# ESM2-3B + ProstT5 extraction into a SEPARATE cache (embeddings_3b/). 64 small
# shards so they cycle through whatever RTX 3080s are free. Bigger model -> smaller
# token budget to fit 10 GB. Merge afterwards: PPI_EMB_DIR=.../embeddings_3b ... merge
set -euo pipefail
NUM_SHARDS=64
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python

export HF_HOME=/home/woody/dsaa/dsaa115h/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# --- 3B config (overrides config.py defaults) ---
export PPI_ESM2_MODEL=/home/woody/dsaa/dsaa115h/hf_cache/esm2_3b_local
export PPI_ESM2_DIM=2560
export PPI_ESM2_MAXTOK=1024
export PPI_EMB_DIR=/home/woody/dsaa/dsaa115h/ppi-entangler/embeddings_3b

cd "$REPO"
mkdir -p logs "$PPI_EMB_DIR"
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} model=$PPI_ESM2_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$PY" dataset.py extract --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}"
