"""
predict_bmse.py — export trained BMSE predictions + penultimate features per pair.

Runs the trained BMSE (best.pt) over every split and dumps, per pair, the
predicted probability and the 64-d penultimate feature vector, so the fusion model
can combine the deep signal with the phylo / structure / coevolution tracks.

Outputs: fusion/bmse_{train,val,test}.npz  (prob, feat [n,64], y, ids)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from dataset import PPIPairDataset
from models import build_model
from train import collate, move, ablate

OUT = Path(__file__).resolve().parent / "fusion"


@torch.no_grad()
def export(model, loader, device, ablation="full"):
    model.eval()
    probs, feats, ys = [], [], []
    for b in loader:
        b = ablate(move(b, device), ablation)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["esm_a"], b["pro_a"], b["mask_a"],
                        b["esm_b"], b["pro_b"], b["mask_b"], b["pool_a"], b["pool_b"])
        probs.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
        feats.append(out["feat"].float().cpu().numpy())
        ys.append(b["y"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(feats, 0), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.RUN_DIR / "bmse" / "best.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    ck = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[predict] loaded {a.ckpt} (val={ck.get('val')})", flush=True)

    for split in ("train", "val", "test"):
        ds = PPIPairDataset(split, max_len=a.max_len)
        dl = DataLoader(ds, a.batch, shuffle=False, collate_fn=collate,
                        num_workers=a.workers, pin_memory=True)
        prob, feat, y = export(model, dl, device)
        ids = np.array([(x[0], x[1]) for x in ds.pairs], dtype="S15")
        np.savez(OUT / f"bmse_{split}.npz", prob=prob, feat=feat, y=y, ids=ids)
        print(f"[predict] {split}: prob={prob.shape} feat={feat.shape}", flush=True)


if __name__ == "__main__":
    main()
