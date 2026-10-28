#!/bin/bash
# doc 35 §27: hybrid scaling at K = 50, 100, 200 on TRL2D and MHD_64 (register + matched MLP; seeds 1-3; TEST once).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
until grep -q S25_DONE $S; do sleep 60; done
for K in 50 100 200; do for seed in 1 2 3; do for dir in turbulent_radiative_layer_2D__gate_vanilla_s1337 MHD_64__gate_vanilla_s1337; do for arm in adrc_ryd adrc_mlp; do
  [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_rs_k${K}_dyn_s${seed}.json ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 10; done
  QW_GPU_GB=14 timeout 7200 runs/capped.sh s27 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup \
     --atoms $K --dyn_tether >> runs/alloc/s27.log 2>&1 &
  sleep 20
done; done; done; done; wait; echo "S27_DONE $(date -Is)" >> $S
