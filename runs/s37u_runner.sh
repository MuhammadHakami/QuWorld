#!/bin/bash
# doc 35 §37u: §37t + gradient clipping 1.0, seeds 1-3, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
L=runs/strat/fusion/head_boundary_v4.log; S=runs/alloc/status
for seed in 1 2 3; do
  [ -f runs/strat/fusion/head_boundary_v4_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_CLIP=1.0 STRAT_LABEL_SUFFIX=_v4 QW_GPU_GB=12 timeout 2400 runs/capped.sh s37u 24 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $seed >> $L 2>&1
  grep -q "^S37P_HEAD_DONE seed=$seed" $L || { echo "CRASH s37u seed $seed $(date -Is)" >> $S; exit 1; }
done
echo "S37U_HEADS_DONE $(date -Is)" >> $S
