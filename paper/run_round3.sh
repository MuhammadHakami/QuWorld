#!/bin/bash
# Round 3 — the 30-simulated-qubit claim with statistics.
# 3 seeds x {none, mlp30, mr30hea, mr30ryd}, 5-min wall-clock sprints, serial.
cd "$(dirname "$0")/.."
for seed in 1337 1338 1339; do
  for v in none mlp30 mr30hea mr30ryd; do
    uv run paper/train.py "$v" "$seed" 300 > "runs/round3_${v}_s${seed}.log" 2>&1 \
      || echo "CRASH ${v} s${seed}"
  done
done
echo ROUND3_DONE
