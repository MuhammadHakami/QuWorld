#!/bin/bash
# Wave 1.5 — the decisive fixed-step grid (throughput-deconfounded):
export PATH="$HOME/.local/bin:$PATH"
# 2,000 steps x {cg30, mr30ryd, mlp30, none} x {k1, k4} x seeds {1337,1338,1339}.
# Launch: setsid nohup bash /home/mohhakami/Documents/paper/quworld/paper/run_ag_wave15.sh \
#   > /home/mohhakami/Documents/paper/quworld/runs/ag15_runner.log 2>&1 < /dev/null &
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1
STATUS=runs/ag_status
STEPS=2000
echo "wave15 start $(date -Is) fixed_steps=${STEPS}" >> "$STATUS"
for seed in 1337 1338 1339; do
  for k in 1 4; do
    for v in none mlp30 mr30ryd cg30; do
      log="runs/ag15_${v}_k${k}_s${seed}.log"
      grep -q "val_vrmse_1" "$log" 2>/dev/null && continue
      echo "run15 ${v} k${k} s${seed} $(date -Is)" >> "$STATUS"
      uv run paper/train.py "$v" "$seed" 900 "$STEPS" "$k" > "$log" 2>&1 \
        || echo "CRASH15 ${v} k${k} s${seed}" >> "$STATUS"
    done
  done
done
echo "WAVE15_DONE $(date -Is)" >> "$STATUS"
sync
