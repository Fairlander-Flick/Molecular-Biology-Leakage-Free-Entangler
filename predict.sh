#!/bin/bash
#SBATCH --job-name=ppi-predict
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/predict_%j.out
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/hpc/dsaa/dsaa115h/ppi-entangler
srun /home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python predict_bmse.py
