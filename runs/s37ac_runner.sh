#!/bin/bash
# doc 35 §37ac: v8 heads, seeds 4-6, seeds 1-3, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
L=runs/strat/fusion/head_boundary_v8.log; S=runs/alloc/status
for seed in 4 5 6; do
  [ -f runs/strat/fusion/head_boundary_v8_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_EPOCHS=60 STRAT_SCAL_W=9.142857 STRAT_CLIP=1.0 STRAT_LABEL_SUFFIX=_v8 QW_GPU_GB=20 timeout 5400 runs/capped.sh s37ac 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $seed >> $L 2>&1
  grep -q "^S37P_HEAD_DONE seed=$seed" $L || { echo "CRASH s37ac seed $seed $(date -Is)" >> $S; exit 1; }
done
echo "S37AC_HEADS_DONE $(date -Is)" >> $S
