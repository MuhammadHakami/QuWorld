#!/bin/bash
# Doc 35 §17 (d): FGN with a VALID-tuned spread scale, vanilla-trunk caches, K = 30. Usage: <lane> [lanes=1]
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; lane=$1; N=${2:-1}; i=0
for dir in turbulent_radiative_layer_2D__gate_vanilla_s1337 shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337 MHD_64__gate_vanilla_s1337; do
  for seed in 1 2 3; do for arm in adrc_ryd adrc_mlp; do echo "$dir $arm $seed"; done; done; done | while read dir arm seed; do
  i=$((i+1)); [ $((i % N)) -ne $lane ] && continue
  f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs_dyn_fgnt_s${seed}.json; [ -f $f ] && continue
  while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge 3 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 22 ]; do sleep 20; done
  QW_GPU_GB=10 timeout 2400 runs/capped.sh fgnt$lane 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether --fgn --fgn_tau >> runs/alloc/fgnt_$lane.log 2>&1 || echo "CRASH fgnt $dir $arm s$seed $(date -Is)" >> $S
done
echo "FGNT_LANE_DONE $lane $(date -Is)" >> $S
