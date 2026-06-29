#!/bin/bash
# watch_regime.sh — block until the C1/C2 regime jobs leave the SLURM queue, then
# dump the regime table. Run in the background; when it exits it re-invokes the
# agent, which continues with the next step (tune/ensemble the C2 point).
set -uo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
cd "$REPO"
J1=$(cat .c1_jid); J2=$(cat .c2_jid)
echo "[watch] waiting on C1=$J1 C2=$J2 ..."
while true; do
  q=$(squeue -u dsaa115h -h -o "%i" 2>/dev/null)
  a=$(echo "$q" | grep -cx "$J1"); b=$(echo "$q" | grep -cx "$J2")
  [ "$a" = "0" ] && [ "$b" = "0" ] && break
  sleep 120
done
echo "[watch] both jobs left the queue at $(date)"
for tag in c1 c2; do
  if [ -f "runs/$tag/test_metrics.json" ]; then
    echo "[watch] $tag DONE -> runs/$tag/test_metrics.json"
  else
    echo "[watch] $tag MISSING test_metrics.json — check logs (may have died/walltimed)"
  fi
done
echo "===== REGIME TABLE ====="
"$PY" regime_curve.py 2>&1
echo "===== END ====="
