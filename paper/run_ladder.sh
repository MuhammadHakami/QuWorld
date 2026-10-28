#!/bin/bash
# Qubit-count scaling ladder (WM-PAI, 2026-09-06): {ryd5, mr10ryd, mr20ryd}
export PATH="$HOME/.local/bin:$PATH"
# x k=4 x seeds {1337,1338,1339} at the exact wave-1.5 protocol (2,000 fixed
# steps) so points pair with existing mr30ryd/none/mlp30/cg30 k4 results.
# Waits for the in-flight MHD mlp30 arm to free the GPU before starting.
# Launch: setsid nohup bash /home/mohhakami/Documents/paper/quworld/paper/run_ladder.sh \
#   > /home/mohhakami/Documents/paper/quworld/runs/ladder_runner.log 2>&1 < /dev/null &
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1
STATUS=runs/ladder_status
STEPS=2000
echo "ladder start $(date -Is) fixed_steps=${STEPS}" >> "$STATUS"
until grep -q "target_reached" runs/phase_b_mhd/ft_mlp30_s1337.log 2>/dev/null; do
  sleep 30
done
echo "gpu free (mlp30 done) $(date -Is)" >> "$STATUS"
for seed in 1337 1338 1339; do
  for v in ryd5 mr10ryd mr20ryd; do
    log="runs/ladder_${v}_k4_s${seed}.log"
    grep -q "val_vrmse_8" "$log" 2>/dev/null && continue
    avail=$(free -g | awk 'NR==2{print $7}')
    [ "$avail" -lt 25 ] && { echo "MEMORY_ABORT ${v} s${seed} $(date -Is)" >> "$STATUS"; sleep 120; }
    echo "run ${v} k4 s${seed} $(date -Is)" >> "$STATUS"
    uv run paper/train.py "$v" "$seed" 1500 "$STEPS" 4 > "$log" 2>&1 \
      || echo "CRASH ${v} k4 s${seed} $(date -Is)" >> "$STATUS"
  done
done
echo "LADDER_DONE $(date -Is)" >> "$STATUS"
sync
