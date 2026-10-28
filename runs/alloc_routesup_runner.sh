#!/bin/bash
# Doc 35 §9: utilization-supervised router on the gated stacks (shear full TEST, RB stratified), 4 cores x 3 seeds.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for dir in rayleigh_benard__gatest_mr30ryd_s1337 shear_flow__gate_mr30ryd_s1337; do
  for seed in 1 2 3; do for arm in adrc_ryd adrc_gnn adrc_mlp adrc_gutz; do
    [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_rs_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 7200 runs/capped.sh rsup 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup >> runs/alloc/adrc.log 2>&1 || echo "CRASH rsup $dir $arm s$seed $(date -Is)" >> $S
  done; done
  echo "RSUP_DONE $dir $(date -Is)" >> $S
done
