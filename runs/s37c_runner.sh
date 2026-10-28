#!/bin/bash
# doc 35 §37c: matched Fusion retrains (old / ip / iplag x seeds 1, 2); all remaining arms run in ONE GPU-lock hold
# (each ~3 min, so other lock users wait at most ~15 min), then CPU scoring on the stratified VALID sample.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; L=runs/strat/fusion/s37c.log; S=runs/alloc/status
[ -f runs/strat/fusion/X_magfix.npy ] || { echo "CRASH s37c magfix missing $(date -Is)" >> $S; exit 1; }
flock runs/strat/gpu.lock bash -c 'until [ $(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo) -ge 30 ]; do sleep 60; done
  for seed in 1 2; do for arm in old ip iplag; do
    [ -f runs/strat/fusion/fix_${arm}_s${seed}.pt ] && continue
    QW_GPU_GB=12 timeout 1200 runs/capped.sh s37c 20 .venv/bin/python paper/strat/fusion/retrain_fix.py --arm $arm --seed $seed \
      || echo "CRASH s37c $arm s$seed $(date -Is)" >> runs/alloc/status
  done; done' >> $L 2>&1
timeout 5400 choom -n 800 -- .venv/bin/python paper/strat/fusion/score_fix.py >> $L 2>&1 && echo "S37C_DONE $(date -Is)" >> $S || echo "CRASH s37c scoring $(date -Is)" >> $S
