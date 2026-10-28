#!/bin/bash
# doc 35 §28c: Shapley credit + handoff under hardware noise (TEST once; fits on VALID), K = 30, seeds 1-3.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
until grep -q S28C_SMOKE_OK $S; do sleep 60; done
for seed in 1 2 3; do for dir in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do for arm in adrc_ryd adrc_mlp; do
  [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_rs_dyn_credit_s${seed}.json ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 1 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 10; done
  QW_GPU_GB=14 timeout 7200 runs/capped.sh s28c 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup \
     --dyn_tether --credit >> runs/alloc/s28c.log 2>&1 &
  sleep 20
done; done; done; wait; echo "S28C_DONE $(date -Is)" >> $S
