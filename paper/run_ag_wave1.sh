#!/bin/bash
# Adaptive-geometry wave 1 (E1 + E4), GPU-concurrent with the overnight campaign.
export PATH="$HOME/.local/bin:$PATH"
# Launch: setsid nohup bash /home/mohhakami/Documents/paper/quworld/paper/run_ag_wave1.sh \
#   > /home/mohhakami/Documents/paper/quworld/runs/ag_runner.log 2>&1 < /dev/null &
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1
STATUS=runs/ag_status
echo "wave1 start $(date -Is)" >> "$STATUS"

# E1: conditioned vs static geometry vs control (one-step objective)
# E4: objective fix isolated (rollout_k=4) incl. classical arms
JOBS="cg30:1 mr30ryd:1 mlp30:1 none:4 mr30ryd:4 mlp30:4 cg30:4 kmix30:1 cg30-3d:1"
for job in $JOBS; do
  v=${job%%:*}; k=${job##*:}
  log="runs/ag1_${v}_k${k}_s1337.log"
  grep -q "val_vrmse_1" "$log" 2>/dev/null && { echo "skip ${v} k${k}" >> "$STATUS"; continue; }
  echo "run ${v} k${k} $(date -Is)" >> "$STATUS"
  uv run paper/train.py "$v" 1337 300 1000000000 "$k" > "$log" 2>&1 \
    || echo "CRASH ${v} k${k}" >> "$STATUS"
done
echo "WAVE1_DONE $(date -Is)" >> "$STATUS"
sync
