#!/bin/bash
# doc 35 §23: TRL2D arena (FNO trunk, 2000 steps, 900 s cap, TD k=4 as the audit arena), 3 seeds; checkpoints + dumps.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 ARENA_SAVE=runs/arena_s23; S=runs/alloc/status
mkdir -p runs/arena_s23; until grep -q S23_TORAX_DONE $S; do sleep 30; done
for seed in 1337 1338 1339; do for v in none q30-geo=3d q30-geo=3d-mlp; do
  [ -f runs/arena_s23/${v}_s${seed}.pt ] && continue
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 24 ]; do sleep 30; done
  QW_GPU_GB=10 timeout 3600 runs/capped.sh s23ar 24 .venv/bin/python paper/train.py "$v" $seed 900 2000 4 > runs/arena_s23/${v}_s${seed}.log 2>&1 \
     || echo "CRASH s23 arena $v s$seed $(date -Is)" >> $S
done; done; echo "S23_ARENA_DONE $(date -Is)" >> $S
