"""
train.py — rigorous HPC training for the BMSE PPI model.

Trains the Bilingual Multi-Scale Entangler over cached ESM2/ProstT5 embeddings on
the leakage-free Bernett splits (train=Intra1, val=Intra0, test=Intra2).

Key ingredients:
  * bf16 autocast (Ampere-native, no GradScaler) + TF32 + optional torch.compile.
  * AdamW, warmup -> cosine LR, early stopping on validation AUPRC.
  * Contrastive data-perturbation (C3PI-inspired): segment-shuffle of the cached
    residue embeddings produces an order-perturbed view; an NT-Xent term pulls the
    two views of a pair together, enforcing sequence-order-invariant learning.
  * Degree-debiasing check: logs correlation between predicted prob and protein
    node degree (the dominant non-biological shortcut in human PPI benchmarks).

Run under SLURM (a100 preferred, rtx3080 fallback) via train.sh.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as C
from dataset import PPIPairDataset
from models import build_model


# --------------------------------------------------------------------------- #
# Collate: pad variable-length chains, build masks
# --------------------------------------------------------------------------- #
def _pad(seqs, dim):
    L = max(s.shape[0] for s in seqs)
    out = torch.zeros(len(seqs), L, dim, dtype=torch.float32)
    mask = torch.zeros(len(seqs), L)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = 1
    return out, mask


def collate(batch):
    ea, ma = _pad([b["esm_a"] for b in batch], C.ESM2_DIM)
    pa, _ = _pad([b["pro_a"] for b in batch], C.PROSTT5_DIM)
    eb, mb = _pad([b["esm_b"] for b in batch], C.ESM2_DIM)
    pb, _ = _pad([b["pro_b"] for b in batch], C.PROSTT5_DIM)
    return {
        "esm_a": ea, "pro_a": pa, "mask_a": ma,
        "esm_b": eb, "pro_b": pb, "mask_b": mb,
        "pool_a": torch.stack([b["pool_a"] for b in batch]).float(),
        "pool_b": torch.stack([b["pool_b"] for b in batch]).float(),
        "y": torch.stack([b["y"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# Segment-shuffle augmentation (on padded residue tensors, per true length)
# --------------------------------------------------------------------------- #
def segment_shuffle(x, mask, k=4):
    """Permute k contiguous segments of each sequence's valid residues."""
    B, L, D = x.shape
    out = x.clone()
    lengths = mask.sum(1).long()
    for i in range(B):
        n = int(lengths[i])
        if n < k:
            continue
        bounds = torch.linspace(0, n, k + 1).long()
        segs = [out[i, bounds[j]:bounds[j + 1]] for j in range(k)]
        perm = torch.randperm(k)
        out[i, :n] = torch.cat([segs[p] for p in perm], dim=0)
    return out


def nt_xent(z1, z2, temp=0.2):
    """NT-Xent over 2B projections; positives = (i, i+B)."""
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.t() / temp
    B = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y, p):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, matthews_corrcoef, roc_auc_score)
    yhat = (p >= 0.5).astype(int)
    return {
        "acc": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
    }


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def ablate(batch, mode):
    """Zero a modality for the embedding ablation. pooled layout =
    [esm_mean(1280) esm_max(1280) | pro_mean(1024) pro_max(1024)] -> esm = [:2560]."""
    if mode == "full":
        return batch
    if mode == "esm":          # keep ESM2, drop ProstT5
        for k in ("pro_a", "pro_b"):
            batch[k] = torch.zeros_like(batch[k])
        batch["pool_a"][:, 2560:] = 0; batch["pool_b"][:, 2560:] = 0
    elif mode == "prostt5":    # keep ProstT5, drop ESM2
        for k in ("esm_a", "esm_b"):
            batch[k] = torch.zeros_like(batch[k])
        batch["pool_a"][:, :2560] = 0; batch["pool_b"][:, :2560] = 0
    return batch


def forward_logits(model, b):
    return model(b["esm_a"], b["pro_a"], b["mask_a"],
                 b["esm_b"], b["pro_b"], b["mask_b"], b["pool_a"], b["pool_b"])


@torch.no_grad()
def evaluate(model, loader, device, degree=None, ablation="full"):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = ablate(move(b, device), ablation)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward_logits(model, b)
        ps.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
        ys.append(b["y"].cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = compute_metrics(y, p)
    if degree is not None:
        m["degree_corr"] = float(np.corrcoef(p, degree)[0, 1])
    return m, y, p


def build_degree(dataset):
    from collections import Counter
    c = Counter()
    for a, b, _ in dataset.pairs:
        c[a] += 1; c[b] += 1
    return np.array([c[a] + c[b] for a, b, _ in dataset.pairs], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--contrast", type=float, default=0.2, help="contrastive loss weight")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap pairs/split")
    ap.add_argument("--ablation", choices=["full","esm","prostt5"], default="full")
    ap.add_argument("--out", type=str, default=str(C.RUN_DIR / "bmse"))
    ap.add_argument("--seed", type=int, default=C.SEED, help="ensemble: vary per run")
    ap.add_argument("--dropout", type=float, default=0.1, help="anti-overfit: raise")
    ap.add_argument("--d-model", type=int, default=256)
    a = ap.parse_args()

    C.apply_runtime_flags()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    part = "a100" if "a100" in torch.cuda.get_device_name(0).lower() else "rtx3080" \
        if device == "cuda" else "rtx3080"
    batch = a.batch or C.TRAIN_BATCH.get(part, 96)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} part={part} batch={batch}", flush=True)

    ds = {s: PPIPairDataset(s, max_len=a.max_len, preload=a.preload)
          for s in ("train", "val", "test")}
    if a.limit:
        import random
        rng = random.Random(C.SEED)
        for s in ds:
            ds[s].pairs = rng.sample(ds[s].pairs, min(a.limit, len(ds[s].pairs)))
    deg = {s: build_degree(ds[s]) for s in ("val", "test")}
    loaders = {
        "train": DataLoader(ds["train"], batch, shuffle=True, collate_fn=collate,
                            num_workers=a.workers, pin_memory=C.PIN_MEMORY, drop_last=True),
        "val": DataLoader(ds["val"], batch, shuffle=False, collate_fn=collate,
                          num_workers=a.workers, pin_memory=C.PIN_MEMORY),
        "test": DataLoader(ds["test"], batch, shuffle=False, collate_fn=collate,
                           num_workers=a.workers, pin_memory=C.PIN_MEMORY),
    }

    model = build_model(d_model=a.d_model, p=a.dropout).to(device)
    if a.compile:
        model = torch.compile(model, mode=C.PRECISION.compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(loaders["train"]) * a.epochs
    warm = int(steps * a.warmup)

    def lr_lambda(s):
        if s < warm:
            return s / max(1, warm)
        prog = (s - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    bce = torch.nn.BCEWithLogitsLoss()

    best, best_ep, wait, history = -1.0, -1, 0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for b in loaders["train"]:
            b = ablate(move(b, device), a.ablation)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward_logits(model, b)
                loss = bce(out["logit"], b["y"])
                if a.contrast > 0:
                    ea2 = segment_shuffle(b["esm_a"], b["mask_a"])
                    pa2 = segment_shuffle(b["pro_a"], b["mask_a"])
                    eb2 = segment_shuffle(b["esm_b"], b["mask_b"])
                    pb2 = segment_shuffle(b["pro_b"], b["mask_b"])
                    out2 = model(ea2, pa2, b["mask_a"], eb2, pb2, b["mask_b"],
                                 b["pool_a"], b["pool_b"])
                    loss = loss + a.contrast * nt_xent(out["proj"], out2["proj"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss)
        vm, *_ = evaluate(model, loaders["val"], device, deg["val"], a.ablation)
        rec = {"epoch": ep, "train_loss": tot / len(loaders["train"]),
               "lr": sched.get_last_lr()[0], "time_s": round(time.time() - t0), **{f"val_{k}": v for k, v in vm.items()}}
        history.append(rec)
        print(f"[ep {ep:02d}] loss={rec['train_loss']:.4f} "
              f"val_acc={vm['acc']:.4f} val_mcc={vm['mcc']:.4f} "
              f"val_auroc={vm['auroc']:.4f} val_auprc={vm['auprc']:.4f} "
              f"deg_corr={vm.get('degree_corr',0):.3f} ({rec['time_s']}s)", flush=True)
        json.dump(history, open(outdir / "history.json", "w"), indent=2)

        if vm["auprc"] > best:
            best, best_ep, wait = vm["auprc"], ep, 0
            torch.save({"model": getattr(model, "_orig_mod", model).state_dict(),
                        "epoch": ep, "val": vm}, outdir / "best.pt")
        else:
            wait += 1
            if wait >= a.patience:
                print(f"[train] early stop at ep {ep} (best ep {best_ep})", flush=True)
                break

    # final test with best checkpoint
    ck = torch.load(outdir / "best.pt", map_location=device)
    getattr(model, "_orig_mod", model).load_state_dict(ck["model"])
    tm, y, p = evaluate(model, loaders["test"], device, deg["test"], a.ablation)
    print(f"[TEST] acc={tm['acc']:.4f} f1={tm['f1']:.4f} mcc={tm['mcc']:.4f} "
          f"auroc={tm['auroc']:.4f} auprc={tm['auprc']:.4f} "
          f"deg_corr={tm.get('degree_corr',0):.3f}", flush=True)
    print(f"[TEST] >>> accuracy {tm['acc']:.4f} vs 0.65 barrier <<<", flush=True)
    json.dump({"test": tm, "best_epoch": best_ep, "best_val_auprc": best},
              open(outdir / "test_metrics.json", "w"), indent=2)
    np.savez(outdir / "test_preds.npz", y=y, p=p)


if __name__ == "__main__":
    main()
