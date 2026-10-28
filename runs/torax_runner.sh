#!/bin/bash
# TORAX-Bench adapter arms on GPU (user: GPU only) (fixed protocol: 1000 steps, batch 16, k=4, 3 seeds), seed-outer for early partial results.
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
S=runs/torax/status; mkdir -p runs/torax; echo "torax runner start $(date -Is)" >> "$S"
for seed in 1337 1338 1339; do for v in none mlp30 mr30ryd; do
  f=runs/torax/${v}_k4_s${seed}.json; [ -f "$f" ] && continue
  echo "run $v s$seed $(date -Is)" >> "$S"
  uv run paper/torax_bench/train_torax.py "$v" "$seed" 1000 4 runs/torax > runs/torax/${v}_k4_s${seed}.log 2>&1 || echo "CRASH $v s$seed $(date -Is)" >> "$S"
done; done
echo "TORAX_DONE $(date -Is)" >> "$S"
