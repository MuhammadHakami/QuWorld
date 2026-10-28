#!/bin/bash
# doc 35 §24 screen (VALID only): 7 configurations x 2 rows x 2 seeds, 4 at a time.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_SCREEN=1; S=runs/alloc/status; O=runs/alloc/adrc_screen
declare -A CFG=( [base]="" [H1]="QW_3D=rand" [H2]="QW_SHIFT=1.0" [H3]="QW_TETHER_INIT=0" [H4]="QW_LAYERS=6" [H5]="QW_HYP=64"
                 [H6]="QW_3D=rand QW_SHIFT=1.0 QW_TETHER_INIT=0 QW_HYP=64" )
for seed in 1 2; do for c in base H1 H2 H3 H4 H5 H6; do for dir in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  mkdir -p $O/$c; ls $O/$c/${dir}_adrc_ryd*_s${seed}.json >/dev/null 2>&1 && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 5; done
  env ${CFG[$c]} QW_OUT=$O/$c QW_GPU_GB=10 timeout 2400 runs/capped.sh s24 16 .venv/bin/python paper/alloc/adrc.py $dir --arm adrc_ryd \
     --seed $seed --gate --route_sup --dyn_tether >> runs/alloc/s24.log 2>&1 &
done; done; done; wait; echo "S24_SCREEN_DONE $(date -Is)" >> $S
