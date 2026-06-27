#!/bin/bash
#SBATCH --job-name=ppi-mmseqs
#SBATCH --partition=work
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/mmseqs_%j.out
#
# One mmseqs search of all 11,018 query proteins vs UniRef50 (seqTaxDB), producing
# BOTH outputs Track A needs:
#   hits.tsv  -> per-hit taxonomy for phylogenetic profiling
#   a3m/      -> per-protein MSAs for paired-MSA coevolution
# Compute nodes have AVX2 (login node is SSE4-only) -> use the avx2 mmseqs here.

set -euo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
SW=/home/woody/dsaa/dsaa115h/software/bin
DB=/home/woody/dsaa/dsaa115h/dbs/uniref50
WORK=/home/woody/dsaa/dsaa115h/ppi-entangler/msa
export PATH=$SW:$PATH
THREADS=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"; mkdir -p logs "$WORK"
TMP="${TMPDIR:-$WORK/tmp}/mmseqs.$SLURM_JOB_ID"; mkdir -p "$TMP"

mmseqs createdb "$WORK/queries.fasta" "$WORK/qDB"
mmseqs search "$WORK/qDB" "$DB" "$WORK/resDB" "$TMP" \
    --threads "$THREADS" -s 6.0 --max-seqs 2000 -e 1e-3 --max-accept 2000

# (1) tabular hits with target taxonomy for phylogenetic profiling
mmseqs convertalis "$WORK/qDB" "$DB" "$WORK/resDB" "$WORK/hits.tsv" \
    --threads "$THREADS" \
    --format-output "query,target,taxid,taxname,fident,evalue,qcov,tcov,bits"

# (2) per-protein a3m MSAs for coevolution (query-anchored, taxid in headers)
mmseqs result2msa "$WORK/qDB" "$DB" "$WORK/resDB" "$WORK/msaDB" \
    --threads "$THREADS" --msa-format-mode 5
mmseqs unpackdb "$WORK/msaDB" "$WORK/a3m" --unpack-suffix .a3m --unpack-data-mode 0 || \
    mmseqs unpackdb "$WORK/msaDB" "$WORK/a3m" --unpack-suffix .a3m

echo "[mmseqs] done: $(wc -l < "$WORK/hits.tsv") hits, $(ls "$WORK/a3m" | wc -l) a3m files"
rm -rf "$TMP"
