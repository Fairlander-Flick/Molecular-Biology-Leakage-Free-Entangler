#!/bin/bash
# watch_3b.sh — wait for the three 3B retrains (c3_3b/c1_3b/c2_3b) to leave the
# queue, then print the 650M-vs-3B regime comparison. Exits -> wakes the agent.
set -uo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
cd "$REPO"
J=$(cat .c3_3b_jid .c1_3b_jid .c2_3b_jid | tr '\n' ' ')
echo "[3b] waiting on $J ..."
while true; do
  left=0
  for j in $J; do squeue -u dsaa115h -h -o "%i" | grep -qx "$j" && left=1; done
  [ "$left" = "0" ] && break
  sleep 120
done
echo "[3b] all retrains left queue at $(date)"
"$PY" - <<'EOF'
import json, os
ROWS=[("C1",("runs/c1","runs/c1_3b")),("C2",("runs/c2","runs/c2_3b")),
      ("C3",("runs/bmse2","runs/c3_3b"))]
def g(p):
    f=os.path.join(p,"test_metrics.json")
    return json.load(open(f))["test"] if os.path.exists(f) else None
print("\n| Regime | 650M acc | 3B acc | 650M AUROC | 3B AUROC |")
print("|---|---|---|---|---|")
for code,(a,b) in ROWS:
    ma,mb=g(a),g(b)
    fa=lambda m,k: f"{m[k]:.3f}" if m else "—"
    print(f"| {code} | {fa(ma,'acc')} | {fa(mb,'acc')} | {fa(ma,'auroc')} | {fa(mb,'auroc')} |")
EOF
echo "[3b] DONE"
