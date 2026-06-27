# Track A — paired-MSA coevolution & phylogenetic profiling

Pairwise evolutionary signal the saturated PLM-embedding methods discard — the
principled route past the ~0.65 leakage-free ceiling.

1. `make_queries.py`  -> queries.fasta (11,018 unique proteins).
2. `run_search.sh`    -> mmseqs vs UniRef50 (seqTaxDB) on an AVX2 compute node:
   * hits.tsv  (per-hit taxonomy) for phylogenetic profiling
   * a3m/      (per-protein MSAs)  for coevolution
3. `../coevo/phylo_features.py` -> per-pair phylogenetic-profile features.
4. `../coevo/coevo_features.py` -> per-pair paired-MSA inter-protein coevolution.

Login node is SSE4-only -> DB download uses `mmseqs-sse41`; search uses AVX2 `mmseqs`.
