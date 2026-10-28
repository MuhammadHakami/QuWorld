#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
S=runs/pelzer/status; mkdir -p runs/pelzer; echo "pelzer runner start $(date -Is)" >> "$S"
for seed in 1337 1338 1339; do for v in none mlp30 mr30ryd; do
  f=runs/pelzer/${v}_s${seed}.json; [ -f "$f" ] && continue
  echo "run $v s$seed $(date -Is)" >> "$S"
  uv run paper/geothermal/train_pelzer.py "$v" "$seed" 1500 runs/pelzer > runs/pelzer/${v}_s${seed}.log 2>&1 || echo "CRASH $v s$seed $(date -Is)" >> "$S"
done; done
echo "PELZER_DONE $(date -Is)" >> "$S"
