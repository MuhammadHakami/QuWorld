#!/bin/bash
# doc 35 §37r: dedicated q95/betaN heads, seeds 1-3, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
L=runs/strat/fusion/head_qb.log; S=runs/alloc/status
for seed in 1 2 3; do
  [ -f runs/strat/fusion/head_qb_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env QW_GPU_GB=16 timeout 2400 runs/capped.sh s37r 32 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $seed qb >> $L 2>&1
  grep -q "^S37R_HEAD_DONE seed=$seed" $L || { echo "CRASH s37r seed $seed $(date -Is)" >> $S; exit 1; }
done
echo "S37R_HEADS_DONE $(date -Is)" >> $S
