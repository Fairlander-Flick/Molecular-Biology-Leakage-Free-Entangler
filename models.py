"""
models.py — Bilingual Multi-Scale Entangler (BMSE).

A structure-aware PPI classifier that fuses two protein "languages" (ESM2
evolutionary + ProstT5 structural) per residue and entangles the two chains via
bidirectional residue-to-residue Cross-Chain Attention, alongside a Multi-Scale
CNN branch that reads physical motifs across receptive fields 16 -> 512.

Forward inputs (variable length, padded per batch):
    esm_a  [B, La, 1280]   pro_a [B, La, 1024]   mask_a [B, La]
    esm_b  [B, Lb, 1280]   pro_b [B, Lb, 1024]   mask_b [B, Lb]
    pool_a [B, 4608]       pool_b [B, 4608]       (mean|max sidecar)

Outputs: dict(logit [B], proj [B, P]) — proj feeds the Step-4 contrastive loss.

Symmetry: PPI is undirected, so the interaction head uses order-invariant
features [a; b; |a-b|; a*b] with a+b / a*b symmetric and |a-b| sign-free, and
the two chains share all encoder weights (Siamese).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# --------------------------------------------------------------------------- #
# Bilingual per-residue fusion
# --------------------------------------------------------------------------- #
class BilingualFusion(nn.Module):
    """Fuse ESM2 (1280) + ProstT5 (1024) per residue -> d_model."""

    def __init__(self, d_model: int, esm_dim=C.ESM2_DIM, pro_dim=C.PROSTT5_DIM, p=0.1):
        super().__init__()
        self.esm_norm = nn.LayerNorm(esm_dim)
        self.pro_norm = nn.LayerNorm(pro_dim)
        self.proj = nn.Sequential(
            nn.Linear(esm_dim + pro_dim, d_model),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, esm, pro):
        x = torch.cat([self.esm_norm(esm), self.pro_norm(pro)], dim=-1)
        return self.out_norm(self.proj(x))


# --------------------------------------------------------------------------- #
# Multi-Scale CNN branch (motifs across receptive fields 16 -> 512)
# --------------------------------------------------------------------------- #
class MultiScaleCNNBranch(nn.Module):
    """Parallel dilated Conv1d branches; dilation extends the receptive field
    of small kernels out to ~512 residues. Operates on [B, L, d_model]."""

    def __init__(self, d_model: int, kernels=(3, 9, 17, 33), ch=64, p=0.1):
        super().__init__()
        # dilations chosen so (k-1)*dil ~ up to ~512 receptive field
        dils = (1, 4, 8, 16)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, ch, k, padding=((k - 1) // 2) * d, dilation=d),
                nn.GELU(),
                nn.BatchNorm1d(ch),
            )
            for k, d in zip(kernels, dils)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(ch * len(kernels), d_model, 1),
            nn.GELU(),
            nn.Dropout(p),
        )

    def forward(self, x, mask):
        h = x.transpose(1, 2)                       # [B, d, L]
        h = torch.cat([b(h) for b in self.branches], dim=1)
        h = self.proj(h).transpose(1, 2)            # [B, L, d]
        return x + h * mask.unsqueeze(-1)           # residual, zero on padding


# --------------------------------------------------------------------------- #
# Bidirectional Cross-Chain Attention
# --------------------------------------------------------------------------- #
class CrossChainAttention(nn.Module):
    """One layer: A attends to B and B attends to A (shared weights), each
    followed by a feed-forward block. Maps residue-residue interface signal."""

    def __init__(self, d_model: int, n_heads=8, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(p), nn.Linear(d_model * 2, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def _cross(self, q, kv, kv_mask):
        # key_padding_mask: True = ignore
        pad = ~kv_mask.bool()
        a, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                         key_padding_mask=pad, need_weights=False)
        q = q + a
        return q + self.ff(self.norm_ff(q))

    def forward(self, a, b, mask_a, mask_b):
        a2 = self._cross(a, b, mask_b)
        b2 = self._cross(b, a, mask_a)
        return a2, b2


# --------------------------------------------------------------------------- #
# Masked pooling
# --------------------------------------------------------------------------- #
def masked_mean_max(x, mask):
    m = mask.unsqueeze(-1)
    s = (x * m).sum(1) / m.sum(1).clamp(min=1)
    mx = x.masked_fill(~m.bool(), float("-inf")).max(1).values
    mx = torch.nan_to_num(mx, neginf=0.0)
    return torch.cat([s, mx], dim=-1)               # [B, 2d]


# --------------------------------------------------------------------------- #
# Full BMSE model
# --------------------------------------------------------------------------- #
class BMSE(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_cross=2, p=0.1, proj_dim=128,
                 pooled_dim=2 * (C.ESM2_DIM + C.PROSTT5_DIM)):
        super().__init__()
        self.fusion = BilingualFusion(d_model, p=p)
        self.cnn = MultiScaleCNNBranch(d_model, p=p)
        self.cross = nn.ModuleList(
            [CrossChainAttention(d_model, n_heads, p) for _ in range(n_cross)])

        self.global_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, d_model), nn.GELU())

        # per-chain descriptor: pooled cross-attended residues (2d) + global (d)
        chain_dim = 2 * d_model + d_model
        # interaction features: [a; b; |a-b|; a*b]  (symmetric)
        inter_dim = 4 * chain_dim
        self.head = nn.Sequential(
            nn.LayerNorm(inter_dim),
            nn.Linear(inter_dim, d_model), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
        )
        self.classifier = nn.Linear(d_model // 2, 1)
        self.proj = nn.Sequential(                  # contrastive projection (Step 4)
            nn.Linear(d_model // 2, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim))

    def encode_chain(self, esm, pro, mask):
        x = self.fusion(esm, pro)
        x = self.cnn(x, mask)
        return x

    def forward(self, esm_a, pro_a, mask_a, esm_b, pro_b, mask_b, pool_a, pool_b):
        a = self.encode_chain(esm_a, pro_a, mask_a)
        b = self.encode_chain(esm_b, pro_b, mask_b)
        for layer in self.cross:
            a, b = layer(a, b, mask_a, mask_b)

        da = torch.cat([masked_mean_max(a, mask_a), self.global_proj(pool_a)], -1)
        db = torch.cat([masked_mean_max(b, mask_b), self.global_proj(pool_b)], -1)

        inter = torch.cat([da, db, (da - db).abs(), da * db], dim=-1)
        h = self.head(inter)
        # `feat` = penultimate representation, exported for the fusion model (Track A+B).
        return {"logit": self.classifier(h).squeeze(-1), "proj": self.proj(h), "feat": h}


def build_model(**kw) -> BMSE:
    return BMSE(**kw)


if __name__ == "__main__":   # shape sanity check on random tensors
    torch.manual_seed(0)
    B, La, Lb = 2, 40, 55
    m = build_model()
    n = sum(p.numel() for p in m.parameters())
    print(f"BMSE params: {n/1e6:.2f}M")
    mask_a = torch.ones(B, La); mask_b = torch.ones(B, Lb)
    out = m(torch.randn(B, La, C.ESM2_DIM), torch.randn(B, La, C.PROSTT5_DIM), mask_a,
            torch.randn(B, Lb, C.ESM2_DIM), torch.randn(B, Lb, C.PROSTT5_DIM), mask_b,
            torch.randn(B, 4608), torch.randn(B, 4608))
    print("logit", out["logit"].shape, "proj", out["proj"].shape)
