#!/bin/bash
# Doc 35 §17 control: dressed deterministic ensembles on the vanilla-trunk caches, K = 30. Usage: <lane> [lanes=2] [flag=--dress|--dress_val]
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; lane=$1; N=${2:-2}; FL=${3:---dress}; SX=$(case $FL in --dress_val) echo dressv;; --dress_rel) echo dressr;; *) echo dress;; esac); i=0
for dir in turbulent_radiative_layer_2D__gate_vanilla_s1337 shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337 MHD_64__gate_vanilla_s1337; do
  for seed in 1 2 3; do for arm in adrc_ryd adrc_mlp; do echo "$dir $arm $seed"; done; done; done | while read dir arm seed; do
  i=$((i+1)); [ $((i % N)) -ne $lane ] && continue
  f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs_dyn_${SX}_s${seed}.json; [ -f $f ] && continue
  while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge 3 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 22 ]; do sleep 20; done
  QW_GPU_GB=10 timeout 2400 runs/capped.sh dress$lane 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether $FL >> runs/alloc/${SX}_$lane.log 2>&1 || echo "CRASH dress $dir $arm s$seed $(date -Is)" >> $S
done
echo "DRESS_LANE_DONE $lane $(date -Is)" >> $S
