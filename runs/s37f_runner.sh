#!/bin/bash
# doc 35 §37f: Fusion marker / lag-floor arms (oldk, iplagk, iplagk50 x seeds 1, 2) in ONE GPU-lock hold, then scoring.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; L=runs/strat/fusion/s37f.log; S=runs/alloc/status
flock runs/strat/gpu.lock bash -c 'until [ $(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo) -ge 30 ]; do sleep 60; done
  for seed in 1 2; do for arm in oldk iplagk iplagk50; do
    [ -f runs/strat/fusion/fix_${arm}_s${seed}.pt ] && continue
    QW_GPU_GB=12 timeout 1200 runs/capped.sh s37f 20 .venv/bin/python paper/strat/fusion/retrain_fix.py --arm $arm --seed $seed \
      || echo "CRASH s37f $arm s$seed $(date -Is)" >> runs/alloc/status
  done; done' >> $L 2>&1
timeout 5400 choom -n 800 -- .venv/bin/python paper/strat/fusion/score_fix.py >> $L 2>&1 && echo "S37F_DONE $(date -Is)" >> $S || echo "CRASH s37f scoring $(date -Is)" >> $S
