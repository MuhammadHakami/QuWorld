#!/bin/bash
# doc 35 §22: frozen-geometry control and 3-D (bilayer) configuration, K=30, 3 seeds, shear_flow + rayleigh_benard.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; mkdir -p runs/alloc/adrc_results_s22/{frozen,3d}
for seed in 1 2 3; do for dir in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  for mode in frozen 3d; do
    while [ "$(jobs -rp | wc -l)" -ge 4 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 18 ]; do sleep 5; done
    if [ $mode = frozen ]; then arm=frozen_ryd; env3d=; else arm=adrc_ryd; env3d=1; fi
    QW_3D=$env3d QW_OUT=runs/alloc/adrc_results_s22/$mode QW_GPU_GB=10 timeout 1500 runs/capped.sh s22 16 \
      .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether >> runs/alloc/s22.log 2>&1 &
  done; done; done
wait; echo "S22_DONE $(date -Is)" >> runs/alloc/status
