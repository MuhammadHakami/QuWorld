#!/bin/bash
# doc 35 §37p step 2: direct shape heads, seeds 1-3, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
L=runs/strat/fusion/head_boundary.log; S=runs/alloc/status
for seed in 1 2 3; do
  [ -f runs/strat/fusion/head_boundary_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env QW_GPU_GB=12 timeout 2400 runs/capped.sh s37p 24 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $seed >> $L 2>&1
  grep -q "^S37P_HEAD_DONE seed=$seed" $L || { echo "CRASH s37p seed $seed $(date -Is)" >> $S; exit 1; }
done
echo "S37P_HEADS_DONE $(date -Is)" >> $S
