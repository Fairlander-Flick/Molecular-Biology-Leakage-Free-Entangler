#!/bin/bash
# watch_final.sh — wait for ALL finalization jobs (3B baseline + ensemble seeds +
# anti-overfit) to leave the queue, then print the master results table. Exits ->
# wakes the agent to write the final docs + figures.
set -uo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
cd "$REPO"
J=$(cat .c3_3b_jid .c1_3b_jid .c2_3b_jid .final_jids 2>/dev/null | tr '\n' ' ')
echo "[final] waiting on: $J"
while true; do
  left=0
  for j in $J; do squeue -u dsaa115h -h -o "%i" | grep -qx "$j" && left=1; done
  [ "$left" = "0" ] && break
  sleep 180
done
echo "[final] all jobs done at $(date)"
echo "===== MASTER TABLE ====="
"$PY" finalize.py 2>&1
echo "===== END ====="
