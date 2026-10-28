#!/bin/bash
# Allocation-strategy study grid (33-allocation-strategy-study.md). Usage: runs/alloc_grid_runner.sh <stream-id> <bench...>
# Phase A per bench: 6 strategies x {lin, mlp, ryd} x 3 seeds at K=30, plus oracle diagnostics (1 seed).
# Phase B per bench: K sweep {10,100,300} for pooled/proxy x {mlp, ryd} (1 seed).
# Phase C per bench: controls gutz / ryd2d / mlpX4 x 3 seeds at the VALID-best tile strategy of the ryd core.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=8
id=$1; shift; S=runs/alloc/status
run() { # bench strategy core K seed
  local f=runs/alloc/results/$1/${2}_${3}_K${4}_s${5}.json
  [ -f $f ] && return
  for attempt in 1 2; do
    timeout 3600 runs/capped.sh alloc-$id 10 .venv/bin/python paper/alloc/coproc.py $1 --strategy $2 --core $3 --K $4 --seed $5 \
      >> runs/alloc/grid_$id.log 2>&1 && return
    echo "CRASH $1 $2 $3 K$4 s$5 attempt $attempt $(date -Is)" >> $S; sleep 30
  done
}
for b in "$@"; do
  while [ ! -f runs/alloc/cache/$b/test.pt ]; do sleep 120; done
  for seed in 1 2 3; do
    for strat in pooled static proxy random learned gain; do
      for core in lin mlp ryd; do run $b $strat $core 30 $seed; done
    done
  done
  for core in lin mlp ryd; do run $b oracle $core 30 1; done
  echo "GRID_A_DONE $b $(date -Is)" >> $S
  for K in 10 100 300; do for strat in pooled proxy; do for core in mlp ryd; do run $b $strat $core $K 1; done; done; done
  echo "GRID_B_DONE $b $(date -Is)" >> $S
  best=$(.venv/bin/python paper/alloc/analyze.py best $b)
  for seed in 1 2 3; do for core in gutz ryd2d mlpX4; do run $b $best $core 30 $seed; done; done
  echo "GRID_C_DONE $b best=$best $(date -Is)" >> $S
done
echo "STREAM_DONE $id $(date -Is)" >> $S
