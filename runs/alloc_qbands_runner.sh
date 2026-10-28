#!/bin/bash
# Doc 35 §10: band-constrained qubit routing (b1-b2) on the gated RB (stratified) and shear (full TEST) stacks.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for seed in 1 2 3; do for dir in rayleigh_benard__gatest_mr30ryd_s1337 shear_flow__gate_mr30ryd_s1337; do for arm in adrc_ryd adrc_gnn adrc_mlp; do
  [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_rs_qb12_s${seed}.json ] && continue
  QW_GPU_GB=12 timeout 7200 runs/capped.sh qb 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --qubit_bands 1,2 >> runs/alloc/adrc.log 2>&1 || echo "CRASH qb $dir $arm s$seed $(date -Is)" >> $S
done; done; done
echo "QB_DONE $(date -Is)" >> $S
