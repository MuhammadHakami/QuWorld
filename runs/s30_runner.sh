#!/bin/bash
# doc 35 §30: proper energy score (fixed TRAIN scale) for the FGN claim; 2 rows x 4 arms x 3 seeds, TEST once.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; O=runs/alloc/adrc_results_s30
for seed in 1 2 3; do for dir in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  for spec in "old adrc_ryd --fgn" "proper adrc_ryd --fgn" "proper adrc_mlp --fgn" "det adrc_ryd --dress_rel"; do
    set -- $spec; tag=$1; arm=$2; shift 2
    out=$O/$tag; mkdir -p $out
    ls $out/${dir}_${arm}_*_s${seed}.json >/dev/null 2>&1 && continue
    while [ "$(jobs -rp | wc -l)" -ge 1 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 15; done
    env $([ $tag = proper ] && echo QW_ES_NORM=train) QW_OUT=$out QW_GPU_GB=14 timeout 10800 runs/capped.sh s30 24 .venv/bin/python paper/alloc/adrc.py \
      $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether "$@" >> runs/alloc/s30.log 2>&1 &
    sleep 20
  done
done; done; wait; echo "S30_DONE $(date -Is)" >> $S
