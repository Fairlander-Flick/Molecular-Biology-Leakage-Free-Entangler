#!/bin/bash
#SBATCH --job-name=ppi-mmseqs
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/mmseqs_%j.out
# (TinyGPU requires a GPU allocation even for CPU-only mmseqs; the GPU is idle here.)
#
# mmseqs search of all 11,018 query proteins vs a target seqTaxDB, producing both
# Track-A inputs:  hits.tsv (taxonomy -> phylo profiling) and a3m/ (-> coevolution).
# Parameterized by env so the same job runs vs Swiss-Prot (fast) or UniRef50 (deep):
#   DB=/path/to/db  TAG=sp|uniref50  sbatch msa/run_search.sh
# Compute nodes have AVX2 (login is SSE4-only) -> use the avx2 mmseqs here.

set -euo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
SW=/home/woody/dsaa/dsaa115h/software/bin
DB="${DB:-/home/woody/dsaa/dsaa115h/dbs/uniref50}"
TAG="${TAG:-uniref50}"
WORK=/home/woody/dsaa/dsaa115h/ppi-entangler/msa/$TAG
export PATH=$SW:$PATH
THREADS=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"; mkdir -p logs "$WORK"
TMP="${TMPDIR:-$WORK}/mmseqs.$SLURM_JOB_ID"; mkdir -p "$TMP"
[ -f "$WORK/qDB" ] || mmseqs createdb "$REPO/msa/queries.fasta" "$WORK/qDB"

mmseqs search "$WORK/qDB" "$DB" "$WORK/resDB" "$TMP" \
    --threads "$THREADS" -s 6.0 --max-seqs 2000 -e 1e-3 --max-accept 2000

mmseqs convertalis "$WORK/qDB" "$DB" "$WORK/resDB" "$WORK/hits.tsv" \
    --threads "$THREADS" \
    --format-output "query,target,taxid,taxname,fident,evalue,qcov,tcov,bits"

mmseqs result2msa "$WORK/qDB" "$DB" "$WORK/resDB" "$WORK/msaDB" \
    --threads "$THREADS" --msa-format-mode 5
rm -rf "$WORK/a3m"
mmseqs unpackdb "$WORK/msaDB" "$WORK/a3m" --unpack-suffix .a3m --unpack-data-mode 0 || \
    mmseqs unpackdb "$WORK/msaDB" "$WORK/a3m" --unpack-suffix .a3m

echo "[mmseqs:$TAG] done: $(wc -l < "$WORK/hits.tsv") hits, $(ls "$WORK/a3m" | wc -l) a3m"
rm -rf "$TMP"
