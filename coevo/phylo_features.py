"""
phylo_features.py — phylogenetic-profile features for each PPI pair (Track A).

Proteins that are gained/lost together across species tend to physically interact
or co-function (classic, homology-leakage-resistant signal orthogonal to PLM
embeddings). From the mmseqs hits we build, per protein, a presence vector over
taxa, TF-IDF-weight it (rare shared taxa matter more), and score each pair by the
similarity of the two profiles.

Inputs : msa/hits.tsv  (query,target,taxid,taxname,fident,evalue,qcov,tcov,bits)
Outputs: coevo/phylo_{train,val,test}.npz  (X [n_pairs, F], y, ids, feat_names)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as D

MSA = Path(__file__).resolve().parents[1] / "msa"
OUT = Path(__file__).resolve().parent
FEATURES = ["n_a", "n_b", "n_shared", "jaccard", "overlap", "cosine_idf",
          "dot_idf", "pearson", "cond_ab", "cond_ba"]


def load_profiles(qcov_min=0.5, eval_max=1e-3):
    """protein -> set of taxids passing thresholds."""
    prof: dict[str, set] = {}
    with open(MSA / "hits.tsv") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            q, taxid, evalue, qcov = p[0], p[2], float(p[5]), float(p[6])
            if taxid in ("", "0") or evalue > eval_max or qcov < qcov_min:
                continue
            prof.setdefault(q, set()).add(int(taxid))
    return prof


def build_matrix(prof, proteins):
    taxa = sorted({t for p in proteins if p in prof for t in prof[p]})
    tax_idx = {t: i for i, t in enumerate(taxa)}
    rows, cols = [], []
    for i, p in enumerate(proteins):
        for t in prof.get(p, ()):
            rows.append(i); cols.append(tax_idx[t])
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(proteins), len(taxa)), dtype=np.float32)
    df = np.asarray(M.sum(0)).ravel()                     # taxon document freq
    idf = np.log((len(proteins) + 1) / (df + 1)) + 1.0
    W = M.multiply(idf).tocsr()                           # tf-idf weighted
    return M, W, idf


def pair_features(M, W, idf, p2i, pairs):
    X, y, ids = [], [], []
    Wn = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel()) + 1e-9
    for a, b, lab in pairs:
        ia, ib = p2i.get(a), p2i.get(b)
        if ia is None or ib is None:
            sa = sb = 0
            row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            ta = set(M.indices[M.indptr[ia]:M.indptr[ia + 1]])
            tb = set(M.indices[M.indptr[ib]:M.indptr[ib + 1]])
            sa, sb = len(ta), len(tb)
            inter = ta & tb; uni = ta | tb
            ns = len(inter)
            jac = ns / max(1, len(uni))
            ov = ns / max(1, min(sa, sb))
            dot = float(W[ia].multiply(W[ib]).sum())
            cos = dot / (Wn[ia] * Wn[ib])
            # pearson over union support
            if uni:
                idx = np.array(sorted(uni))
                va = np.array([1.0 if t in ta else 0.0 for t in idx])
                vb = np.array([1.0 if t in tb else 0.0 for t in idx])
                pr = float(np.corrcoef(va, vb)[0, 1]) if va.std() and vb.std() else 0.0
            else:
                pr = 0.0
            row = [sa, sb, ns, jac, ov, cos, dot, pr,
                   ns / max(1, sa), ns / max(1, sb)]
        X.append(row); y.append(lab); ids.append((a, b))
    return np.array(X, np.float32), np.array(y, np.int8), ids


def main():
    pairs_by_split, _, _ = D.build_manifest()
    proteins = sorted({p for v in pairs_by_split.values() for a, b, _ in v for p in (a, b)})
    p2i = {p: i for i, p in enumerate(proteins)}
    prof = load_profiles()
    print(f"[phylo] {len(prof)} proteins with hits; {len(proteins)} in pairs", flush=True)
    M, W, idf = build_matrix(prof, proteins)
    print(f"[phylo] taxon space: {M.shape[1]}", flush=True)
    for split, pairs in pairs_by_split.items():
        X, y, ids = pair_features(M, W, idf, p2i, pairs)
        np.savez(OUT / f"phylo_{split}.npz", X=X, y=y,
                 ids=np.array(ids, dtype="S15"), feat_names=np.array(FEATURES))
        print(f"[phylo] {split}: X={X.shape} pos_rate={y.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
