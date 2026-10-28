#!/bin/bash
# Division-of-labor cores (doc 33): hyb (register + matched MLP on the same atoms) vs mlp2 (two matched MLPs), 3 seeds,
# at each row's VALID-best tile strategy of the register, after that row's phase C. Usage: runs/alloc_extra_runner.sh <bench...>
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=8; S=runs/alloc/status
for b in "$@"; do
  until grep -q "GRID_C_DONE $b " $S; do sleep 120; done
  best=$(.venv/bin/python paper/alloc/analyze.py best $b)
  for seed in 1 2 3; do for core in hyb mlp2; do
    f=runs/alloc/results/$b/${best}_${core}_K30_s${seed}.json; [ -f $f ] && continue
    timeout 3600 runs/capped.sh alloc-x 10 .venv/bin/python paper/alloc/coproc.py $b --strategy $best --core $core --K 30 --seed $seed >> runs/alloc/grid_X.log 2>&1
  done; done
  echo "PAIR_DONE $b best=$best $(date -Is)" >> $S
done
