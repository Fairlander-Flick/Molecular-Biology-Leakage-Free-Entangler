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
HF_HOME = WOODY / "hf_cache"   # pre-staged ESM2/ProstT5 (compute nodes are offline)

DATA_DIR = WOODY / "ppi-entangler/data"          # raw Bernett dataset
EMB_DIR = Path(os.environ.get("PPI_EMB_DIR", str(WOODY / "ppi-entangler/embeddings")))
CKPT_DIR = WOODY / "ppi-entangler/checkpoints"
RUN_DIR = PROJECT_ROOT / "runs"

# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"   # 650M, 33 layers, dim 1280
# ProstT5 ships only pytorch_model.bin; transformers 5.12 refuses torch.load on
# torch<2.6. We pre-convert to safetensors into a standalone local dir and load
# from there (avoids HF offline-cache resolution of the manually-added file).
PROSTT5_MODEL = os.environ.get("PROSTT5_PATH", str(WOODY / "hf_cache/prostt5_local"))
ESM2_DIM = 1280
PROSTT5_DIM = 1024
# Two-tier length policy: embed full up to the cap; longer proteins -> head512+tail512.
# Cap 1024 covers full length for 85.7% of proteins (vs 55.7% at 512).
MAX_SEQ_LEN_EMB = 1024   # per-residue embedding cap at extraction time
MAX_SEQ_LEN = 1024       # per-residue length fed to the model (cross-attention)
CNN_SCALE_MIN = 16       # multi-scale CNN spans receptive fields 16 -> 512
CNN_SCALE_MAX = 512


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
# Length-bucketed extraction: token budget per forward pass (sum of residues in a
# batch). ProstT5 (~3B) is heavier than ESM2 (650M); tuned conservative for 10 GB.
EXTRACT_MAX_TOKENS = {"esm2": 4096, "prostt5": 1024}

TRAIN_BATCH = {    # pair-samples per step. Cross-chain attention is O(B*La*Lb),
    "a100": 48,    # so memory is dominated by per-residue length, not the head.
    "rtx3080": 12, # dynamic length-padding keeps short-protein batches cheap.
    "v100": 24,
}

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
PIN_MEMORY = True
HDF5_CHUNK = 64   # chunked HDF5 rows for streaming + pinned-memory transfer

SEED = 42
