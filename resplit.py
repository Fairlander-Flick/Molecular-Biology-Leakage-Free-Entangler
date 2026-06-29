"""resplit.py — build C1/C2/C3 leakage-regime splits from the pooled Bernett edges.

The Bernett gold-standard ships three protein-disjoint partitions (Intra0/1/2);
our default pipeline uses them as the strict C3 regime (train/test proteins
disjoint -> both endpoints novel). Here we pool ALL edges + proteins and
re-partition them into the three canonical PPI-evaluation regimes so we can
report accuracy as a function of leakage on ONE dataset and ONE embedding cache:

  C1  random EDGE split          -> test edges' endpoints almost all seen in
                                    train (both-seen). Proof-of-mechanism.
  C2  random NOVEL-PROTEIN subset -> train = edges with both endpoints seen;
                                    test = edges with exactly one endpoint novel.
  C3  = the original Bernett strict split (both endpoints novel). Not rebuilt
        here; reuse runs/bmse2 (acc 0.660).

Writes {OUT}/c1/{train,val,test}.tsv and {OUT}/c2/{...}. Each line: "a b y".
Sizes mirror the original split (train 163192 / val 59260 / test 52048) so the
three regimes are size-comparable.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import config as C
from dataset import read_fasta, read_pairs

SEED = 1234
OUT = C.PROJECT_ROOT / "splits"
# Original Bernett split sizes (Intra1 / Intra0 / Intra2), for size parity.
N_TRAIN, N_VAL, N_TEST = 163192, 59260, 52048


def write_split(d: Path, train, val, test):
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with open(d / f"{name}.tsv", "w") as fh:
            for a, b, y in rows:
                fh.write(f"{a}\t{b}\t{y}\n")
    bal = lambda rows: dict(Counter(y for *_, y in rows))
    print(f"  {d.name}: train={len(train)} {bal(train)} | "
          f"val={len(val)} {bal(val)} | test={len(test)} {bal(test)}", flush=True)


def main():
    rng = random.Random(SEED)
    seqs = read_fasta(C.DATA_DIR)

    # Pool every edge across the three partitions, dropping missing-seq pairs.
    edges = []
    for part in ("Intra0", "Intra1", "Intra2"):
        for a, b, y in read_pairs(part):
            if a in seqs and b in seqs:
                edges.append((a, b, y))
    proteins = sorted({p for a, b, _ in edges for p in (a, b)})
    print(f"[pool] {len(edges)} edges | {len(proteins)} proteins", flush=True)

    # ---- C1: random edge split (both endpoints almost surely seen) ----------
    e = edges[:]
    rng.shuffle(e)
    c1_train = e[:N_TRAIN]
    c1_val = e[N_TRAIN:N_TRAIN + N_VAL]
    c1_test = e[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]
    # leakage check: fraction of C1 test edges with BOTH endpoints seen in train
    seen = {p for a, b, _ in c1_train for p in (a, b)}
    both = sum(1 for a, b, _ in c1_test if a in seen and b in seen)
    print(f"[C1] test edges both-seen: {both}/{len(c1_test)} "
          f"({both / len(c1_test):.3f})", flush=True)
    write_split(OUT / "c1", c1_train, c1_val, c1_test)

    # ---- C2: novel-protein holdout (exactly one endpoint novel in test) -----
    # Mark a random ~15% of proteins novel. Train = edges with NEITHER endpoint
    # novel; eval = edges with EXACTLY ONE novel endpoint (both-novel dropped =
    # that is the C3 regime). Split eval into val/test.
    pr = proteins[:]
    rng.shuffle(pr)
    n_novel = int(round(0.15 * len(pr)))
    novel = set(pr[:n_novel])
    c2_train_all, c2_eval, dropped = [], [], 0
    for a, b, y in edges:
        na, nb = a in novel, b in novel
        if not na and not nb:
            c2_train_all.append((a, b, y))
        elif na ^ nb:                     # exactly one novel -> C2 regime
            c2_eval.append((a, b, y))
        else:
            dropped += 1                  # both novel -> C3, drop here
    rng.shuffle(c2_train_all)
    rng.shuffle(c2_eval)
    c2_train = c2_train_all[:N_TRAIN]
    n_val = min(N_VAL, len(c2_eval) // 2)
    c2_val = c2_eval[:n_val]
    c2_test = c2_eval[n_val:n_val + N_TEST]
    print(f"[C2] novel proteins={len(novel)} | train_pool={len(c2_train_all)} "
          f"eval_pool={len(c2_eval)} both-novel-dropped={dropped}", flush=True)
    # leakage check: every C2 test edge must have exactly one endpoint in train-seen
    seen2 = {p for a, b, _ in c2_train for p in (a, b)}
    bad = sum(1 for a, b, _ in c2_test
              if (a in seen2) == (b in seen2))   # want exactly one seen
    print(f"[C2] test edges violating one-novel: {bad}/{len(c2_test)}", flush=True)
    write_split(OUT / "c2", c2_train, c2_val, c2_test)

    print(f"[done] splits written under {OUT}", flush=True)


if __name__ == "__main__":
    main()
