#!/bin/bash
# Doc 35 §13: qubit-count scaling, fully input-dependent configuration. Usage: runs/alloc_scaling_runner.sh <lane 0-3>
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; lane=$1; i=0
jobs() {
  for dir in shear_flow__gate_mr30ryd_s1337 rayleigh_benard__gatest_mr30ryd_s1337; do for seed in 1 2 3; do for k in 5 10 20 30; do for arm in adrc_ryd adrc_mlp; do echo "$dir $arm $seed $k"; done; done; done; done
  for dir in turbulent_radiative_layer_2D__gate_mr30ryd_s1337 MHD_64__gate_vanilla_s1337; do for k in 5 10 20 30; do for arm in adrc_ryd adrc_mlp; do echo "$dir $arm 1 $k"; done; done; done
}
jobs | while read dir arm seed k; do
  i=$((i+1)); [ $((i % 4)) -ne $lane ] && continue
  sfx=$([ $k -ne 30 ] && echo "_k$k"); f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs${sfx}_dyn_s${seed}.json
  [ -f $f ] && continue
  QW_GPU_GB=12 timeout 1800 runs/capped.sh scale$lane 20 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --atoms $k --dyn_tether >> runs/alloc/scaling_$lane.log 2>&1 || echo "CRASH scaling $dir $arm s$seed k$k $(date -Is)" >> $S
done
echo "SCALING_LANE_DONE $lane $(date -Is)" >> $S
