#!/bin/bash
# ADR-C v1 registered runs (doc 35 §6): caches, then 8 arms x 3 seeds per row. Usage: runs/alloc_adrc_runner.sh <row...>
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for r in "$@"; do
  [ -f runs/alloc/adrc_cache/$r/test.pt ] || QW_GPU_GB=24 timeout 7200 runs/capped.sh adrc-c3 32 .venv/bin/python paper/alloc/adrc_cache.py $r >> runs/alloc/adrc_cache.log 2>&1 || echo "CRASH adrc_cache $r $(date -Is)" >> $S
  for seed in 1 2 3; do for arm in cls_only static_lin adrc_ryd adrc_gnn adrc_mlp frozen_ryd adrc_gutz rand_ryd; do
    [ -f runs/alloc/adrc_results/${r}_${arm}_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 5400 runs/capped.sh adrc-rb3 16 .venv/bin/python paper/alloc/adrc.py $r --arm $arm --seed $seed >> runs/alloc/adrc.log 2>&1 || echo "CRASH adrc $r $arm s$seed $(date -Is)" >> $S
  done; done
  echo "ADRC_DONE $r $(date -Is)" >> $S
done
