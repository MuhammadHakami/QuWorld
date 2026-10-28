#!/bin/bash
# doc 35 §37y: v5 heads, seeds 4-6, seeds 1-3, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
L=runs/strat/fusion/head_boundary_v5.log; S=runs/alloc/status
for seed in 4 5 6; do
  [ -f runs/strat/fusion/head_boundary_v5_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_CLIP=1.0 STRAT_LABEL_SUFFIX=_v5 QW_GPU_GB=20 timeout 3000 runs/capped.sh s37y 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $seed >> $L 2>&1
  grep -q "^S37P_HEAD_DONE seed=$seed" $L || { echo "CRASH s37y seed $seed $(date -Is)" >> $S; exit 1; }
done
echo "S37Y_HEADS_DONE $(date -Is)" >> $S
