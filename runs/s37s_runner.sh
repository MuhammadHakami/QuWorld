#!/bin/bash
# doc 35 §37s: 64-ray shape heads on all TRAIN frames, seeds 1-3, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
L=runs/strat/fusion/head_boundary_v2.log; S=runs/alloc/status
for seed in 1 2 3; do
  [ -f runs/strat/fusion/head_boundary_v2_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v2 QW_GPU_GB=20 timeout 3000 runs/capped.sh s37s 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $seed >> $L 2>&1
  grep -q "^S37P_HEAD_DONE seed=$seed" $L || { echo "CRASH s37s seed $seed $(date -Is)" >> $S; exit 1; }
done
echo "S37S_HEADS_DONE $(date -Is)" >> $S
