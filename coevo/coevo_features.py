"""
coevo_features.py — paired-MSA inter-protein coevolution features (Track A).

For each protein we parse its a3m MSA into (taxid -> match-state sequence) and keep
the K most-variable match columns. For a pair we pair rows by taxid (orthologs in
the same species) and measure inter-protein coevolution — APC-corrected mutual
information between A-columns and B-columns — the residue-level signal AlphaFold-
Multimer exploits, plus paired MSA depth (interacting proteins co-occur deeply).

Inputs : msa/a3m/*.a3m  (query-anchored; query id = first record header),
         msa/hits.tsv   (target -> taxid fallback)
Outputs: coevo/coevo_{split}.npz  (X, y, ids, feat_names)  [shardable over pairs]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as D

MSA = Path(os.environ.get("PPI_MSA_DIR",
                          str(Path(__file__).resolve().parents[1] / "msa" / "uniref50")))
OUT = Path(__file__).resolve().parent
TAXID_RE = re.compile(r"TaxID=(\d+)")
K = 24                       # top-variable match columns kept per protein
AA = "ACDEFGHIKLMNPQRSTVWYX-"
A2I = {c: i for i, c in enumerate(AA)}
FEATURES = ["paired_depth", "neff_a", "neff_b", "log_paired_depth",
            "max_apc_mi", "mean_top5_apc_mi", "mean_apc_mi"]


def target_taxid_map():
    m = {}
    f = MSA / "hits.tsv"
    if f.exists():
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] not in m and p[2] not in ("", "0"):
                m[p[1]] = int(p[2])
    return m


def parse_a3m(path, tax_map):
    """Return (taxids[n], match_matrix[n, Lq] uint8) keeping one row per taxid."""
    hdr, seqs = [], []
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                hdr.append(line[1:]); cur = []; seqs.append(cur)
            elif cur is not None:
                cur.append(line)
    if not seqs:
        return None, None
    rows = ["".join(s) for s in seqs]
    # match columns = uppercase + '-' (drop lowercase insertions), anchored to query
    def match_only(s):
        return "".join(ch for ch in s if ch.isupper() or ch == "-")
    M = [match_only(r) for r in rows]
    Lq = len(M[0])
    seen, taxids, mat = set(), [], []
    for h, m in zip(hdr, M):
        if len(m) != Lq:
            continue
        mm = TAXID_RE.search(h)
        tid = int(mm.group(1)) if mm else tax_map.get(h.split()[0])
        if tid is None or tid in seen:
            continue
        seen.add(tid); taxids.append(tid)
        mat.append([A2I.get(c, A2I["X"]) for c in m])
    if len(mat) < 2:
        return None, None
    return np.array(taxids), np.array(mat, np.uint8)


def reduce_columns(mat):
    """Keep K most-variable columns (by entropy)."""
    n, L = mat.shape
    ent = np.zeros(L)
    for j in range(L):
        _, cnt = np.unique(mat[:, j], return_counts=True)
        p = cnt / cnt.sum()
        ent[j] = -(p * np.log(p + 1e-9)).sum()
    cols = np.argsort(ent)[::-1][:K]
    return mat[:, cols]


@njit(parallel=True, cache=True)
def apc_mi(A, B):
    """APC-corrected mutual information matrix between columns of A and B.
    Marginals/means use explicit loops (numba does not accept axis= args)."""
    n, Ka = A.shape
    Kb = B.shape[1]
    mi = np.zeros((Ka, Kb))
    for i in prange(Ka):
        pa = np.zeros(22); pb = np.zeros(22)
        joint = np.zeros((22, 22))
        for j in range(Kb):
            joint[:] = 0.0; pa[:] = 0.0; pb[:] = 0.0
            for r in range(n):
                joint[A[r, i], B[r, j]] += 1.0
            for x in range(22):
                for y in range(22):
                    joint[x, y] /= n
                    pa[x] += joint[x, y]
                    pb[y] += joint[x, y]
            m = 0.0
            for x in range(22):
                for y in range(22):
                    if joint[x, y] > 0 and pa[x] > 0 and pb[y] > 0:
                        m += joint[x, y] * np.log(joint[x, y] / (pa[x] * pb[y]))
            mi[i, j] = m
    # APC correction with explicit means
    ri = np.zeros(Ka); rj = np.zeros(Kb); tot = 0.0
    for i in range(Ka):
        for j in range(Kb):
            ri[i] += mi[i, j]; rj[j] += mi[i, j]; tot += mi[i, j]
    for i in range(Ka):
        ri[i] /= Kb
    for j in range(Kb):
        rj[j] /= Ka
    mbar = tot / (Ka * Kb)
    if mbar > 0:
        for i in range(Ka):
            for j in range(Kb):
                mi[i, j] -= ri[i] * rj[j] / mbar
    return mi


def neff(mat):
    return float(np.log1p(mat.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    pairs_by_split, _, _ = D.build_manifest()
    tax_map = target_taxid_map()

    # parse + reduce all a3m once (cache per protein)
    cache = {}
    for f in (MSA / "a3m").glob("*.a3m"):
        tax, mat = parse_a3m(f, tax_map)
        if tax is None:
            continue
        pid = open(f).readline()[1:].split()[0]
        cache[pid] = (tax, reduce_columns(mat))
    print(f"[coevo] parsed {len(cache)} MSAs", flush=True)

    for split, pairs in pairs_by_split.items():
        sub = pairs[a.shard::a.num_shards]
        X, y, ids = [], [], []
        for pa, pb, lab in sub:
            row = [0.0] * len(FEATURES)
            if pa in cache and pb in cache:
                ta, Ma = cache[pa]; tb, Mb = cache[pb]
                common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
                if len(common) >= 5:
                    mi = apc_mi(Ma[ia], Mb[ib])
                    flat = np.sort(mi.ravel())[::-1]
                    row = [float(len(common)), neff(Ma), neff(Mb),
                           float(np.log1p(len(common))), float(flat[0]),
                           float(flat[:5].mean()), float(mi.mean())]
            X.append(row); y.append(lab); ids.append((pa, pb))
        suff = "" if a.num_shards == 1 else f"_{a.shard:03d}_{a.num_shards:03d}"
        np.savez(OUT / f"coevo_{split}{suff}.npz", X=np.array(X, np.float32),
                 y=np.array(y, np.int8), ids=np.array(ids, dtype="S15"),
                 feat_names=np.array(FEATURES))
        print(f"[coevo] {split}{suff}: X={np.array(X).shape}", flush=True)


if __name__ == "__main__":
    main()
