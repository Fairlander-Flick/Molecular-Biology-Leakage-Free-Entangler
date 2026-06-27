"""
Central hardware-aware configuration for the Leakage-Free Entangler pipeline.

Encodes the Step-1 decisions approved for NHR@FAU TinyGPU:
  * Embedding extraction  -> parallel RTX 3080 SLURM array (8 GPU/node, idle capacity)
  * BMSE training         -> A100 preferred, RTX 3080 fallback (head trains on cached embeddings)
  * Precision             -> bfloat16 autocast + TF32 (Ampere-native, no GradScaler)
  * Compilation           -> torch.compile on the BMSE model; PLM extraction stays eager
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths   (home = code/small; woody = bulk caches; node-local /scratch = staging)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
WOODY = Path("/home/woody/dsaa/dsaa115h")
CONDA_PYTHON = WOODY / "software/private/conda/envs/kaggle/bin/python"

DATA_DIR = WOODY / "ppi-entangler/data"          # raw Bernett dataset
EMB_DIR = WOODY / "ppi-entangler/embeddings"     # chunked HDF5 caches
CKPT_DIR = WOODY / "ppi-entangler/checkpoints"
RUN_DIR = PROJECT_ROOT / "runs"

# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"   # 650M, 33 layers, dim 1280
PROSTT5_MODEL = "Rostlab/ProstT5"             # ~3B, dim 1024 (3Di-aware)
ESM2_DIM = 1280
PROSTT5_DIM = 1024
MAX_SEQ_LEN = 512   # multi-scale CNN spans receptive fields 16 -> 512


# --------------------------------------------------------------------------- #
# SLURM partitions (live scan: see HARDWARE.md)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Partition:
    name: str
    gres: str          # SLURM --gres token, e.g. "gpu:a100:1"
    vram_gb: float
    cc: str            # CUDA compute capability
    gpus_per_node: int
    cpus_per_node: int
    sys_ram_gb: int
    bf16: bool


PARTITIONS = {
    "a100":    Partition("a100",    "gpu:a100:1",      40.0, "8.0", 4, 128, 510, True),
    "v100":    Partition("v100",    "gpu:v100:1",      32.0, "7.0", 4,  32,  95, False),
    "rtx3080": Partition("rtx3080", "gpu:rtx3080:1",   10.0, "8.6", 8,  64, 380, True),
}

EXTRACT_PARTITION = "rtx3080"   # embarrassingly-parallel array on idle nodes
TRAIN_PARTITION = "a100"        # preferred; fall back to rtx3080 if queue-bound
TRAIN_FALLBACK = "rtx3080"


# --------------------------------------------------------------------------- #
# Precision / compilation   (bf16 + TF32 + selective torch.compile)
# --------------------------------------------------------------------------- #
@dataclass
class PrecisionConfig:
    amp_dtype: str = "bfloat16"   # Ampere-native; stable, no overflow scaling
    use_grad_scaler: bool = False # bf16 => GradScaler unnecessary
    allow_tf32: bool = True       # TF32 matmul + cudnn
    compile_model: bool = True    # torch.compile on the BMSE head
    compile_mode: str = "max-autotune"
    compile_plm: bool = False     # PLM extraction stays eager


PRECISION = PrecisionConfig()


def apply_runtime_flags() -> None:
    """Enable TF32 etc. Call once at process start, after `import torch`."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = PRECISION.allow_tf32
    torch.backends.cudnn.allow_tf32 = PRECISION.allow_tf32
    torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------------- #
# Phase-specific batch sizing
#   Extraction is VRAM-bound (PLM weights + L x D per-residue tensors);
#   training runs over small cached embeddings, so batches can be large.
# --------------------------------------------------------------------------- #
EXTRACT_BATCH = {  # sequences per forward pass, keyed by partition
    "rtx3080": {"esm2": 8,  "prostt5": 4},   # 10 GB: chunk long seqs
    "a100":    {"esm2": 64, "prostt5": 32},
    "v100":    {"esm2": 16, "prostt5": 8},
}

TRAIN_BATCH = {    # pair-samples per step (head over cached embeddings)
    "a100": 256,
    "rtx3080": 96,
    "v100": 64,
}

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
PIN_MEMORY = True
HDF5_CHUNK = 64   # chunked HDF5 rows for streaming + pinned-memory transfer

SEED = 42
