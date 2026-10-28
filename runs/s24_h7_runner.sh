#!/bin/bash
# doc 35 §24 addendum: H7 (neighbour-correlator readout), VALID only, 2 rows x 2 seeds, after the main screen.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_SCREEN=1 QW_ZZ=1; S=runs/alloc/status; O=runs/alloc/adrc_screen/H7
until grep -q S24_SCREEN_DONE $S; do sleep 60; done; mkdir -p $O
for seed in 1 2; do for dir in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  ls $O/${dir}_adrc_ryd*_s${seed}.json >/dev/null 2>&1 && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 5; done
  QW_OUT=$O QW_GPU_GB=10 timeout 2400 runs/capped.sh s24h7 16 .venv/bin/python paper/alloc/adrc.py $dir --arm adrc_ryd --seed $seed \
     --gate --route_sup --dyn_tether >> runs/alloc/s24.log 2>&1 &
done; done; wait; echo "S24_H7_DONE $(date -Is)" >> $S
