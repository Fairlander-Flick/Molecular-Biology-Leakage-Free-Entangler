#!/bin/bash
# watch_ext3b.sh — wait for the ESM2-3B extraction array to finish, then merge the
# shards into embeddings_3b/embeddings.h5 and verify protein coverage. Exits when
# done (which re-invokes the agent to launch the 3B C1/C2/C3 retrains).
set -uo pipefail
REPO=/home/hpc/dsaa/dsaa115h/ppi-entangler
PY=/home/woody/dsaa/dsaa115h/software/private/conda/envs/kaggle/bin/python
EMB=/home/woody/dsaa/dsaa115h/ppi-entangler/embeddings_3b
cd "$REPO"
JID=$(cat .ext3b_jid)
echo "[ext3b] waiting on extraction array $JID ..."
while squeue -u dsaa115h -h -o "%i" 2>/dev/null | grep -q "^${JID}"; do sleep 120; done
echo "[ext3b] array left queue at $(date)"
n=$(ls "$EMB"/shard_*.h5 2>/dev/null | wc -l)
echo "[ext3b] shards present: $n / 64"
if [ "$n" -lt 64 ]; then
  echo "[ext3b] MISSING shards — NOT merging. Re-run failed array indices."
  echo "[ext3b] present indices:"; ls "$EMB"/shard_*.h5 2>/dev/null | sed 's/.*shard_0*//;s/_.*//' | sort -n | tr '\n' ' '
  exit 1
fi
echo "[ext3b] all shards present -> merging"
export HF_HUB_OFFLINE=1 PPI_ESM2_DIM=2560 PPI_EMB_DIR="$EMB"
"$PY" dataset.py merge 2>&1 | tail -5
"$PY" - <<'EOF'
import h5py, os
p=os.path.join("/home/woody/dsaa/dsaa115h/ppi-entangler/embeddings_3b","embeddings.h5")
with h5py.File(p) as f:
    print(f"[ext3b] merged cache: {len(f['ids'])} proteins | esm2 dim {f['esm2'].shape[1]} | residues {f['esm2'].shape[0]}")
EOF
echo "[ext3b] DONE — ready for 3B retrains"
