# Hardware Scan — NHR@FAU TinyGPU

Live SLURM/CUDA scan performed 2026-06-27. Reproduce with `gpu_profile.py` under SLURM:

```bash
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
srun --partition=rtx3080 --gres=gpu:rtx3080:1 --time=00:03:00 --cpus-per-task=8 $PY gpu_profile.py
```

## Cluster

- **Login node:** `tinyx` (no GPU; all compute via SLURM).
- **Scheduler:** SLURM (`sinfo`/`sbatch`/`srun`/`salloc`), account `dsaa`, QOS `normal`.
- **Storage:** `/home/hpc` (NFS, shared, code), `/home/woody` (1.6 PB, bulk caches),
  node-local `/scratch` (1.7 TB SSD, staging — **not** visible from login node).
- **Toolchain:** conda env `kaggle` → torch **2.5.1+cu121**, transformers **5.12.0**;
  driver 610.43.02 / CUDA UMD 13.3; Apptainer/Singularity available.

## GPU partitions

| Partition | GPUs/node | VRAM | Compute Cap. | bf16 | Sys RAM | CPUs | Time limit |
|-----------|-----------|------|--------------|------|---------|------|------------|
| `a100`    | 4× A100   | ~40 GB *(probe queued — backfill)* | 8.0 | ✅ | 510 GB | 128 | 1 day |
| `v100`    | 4× V100   | 16/32 GB | 7.0 | ❌ | 95 GB | 32 | 1 day |
| `rtx3080` | 8× RTX 3080 | **10 GB** (confirmed `tg080`) | 8.6 | ✅ | 380 GB | 64 | 1 day |
| `work*` (default) | 8× RTX 3080 / 4× RTX 2080 Ti | 10 / 11 GB | 8.6 / 7.5 | ✅/❌ | 380/95 GB | 64/32 | 1 day |

Confirmed on `tg080`: `NVIDIA GeForce RTX 3080`, 10240 MiB, driver 610.43.02.

## Availability at scan time

- `a100`: **fully allocated** (`alloc`); probe queued — initially `(Priority)`, then `(AssocGrpGRES)` group GPU cap.
- `v100`: fully allocated.
- `rtx3080`: **13 idle nodes** (5 idle + 1 mix in `rtx3080`, plus default `work` pool) → ideal for a parallel extraction array.

## Approved hardware-aware strategy (Step 1)

1. **Extraction** → RTX 3080 SLURM **array** (8 GPU/node × idle nodes), bf16 inference, long-seq chunking for 10 GB.
2. **Training** → **A100** preferred (bf16, TF32, large batch), **RTX 3080 fallback** — the BMSE head
   trains over *cached* embeddings and is not VRAM-bound.
3. **Precision** → `bfloat16` autocast (no `GradScaler`) + TF32 matmul.
4. **Compilation** → `torch.compile(mode="max-autotune")` on BMSE; PLM extraction eager.

Encoded in [`config.py`](config.py).
