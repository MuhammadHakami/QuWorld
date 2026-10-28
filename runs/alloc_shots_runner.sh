#!/bin/bash
# Doc 35 §11: eval-only finite-shot readout on the route-supervised register (retrained as in §9, same seeds).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
for dir in shear_flow__gate_mr30ryd_s1337 rayleigh_benard__gatest_mr30ryd_s1337; do
  for seed in 1 2 3; do
    [ -f runs/alloc/adrc_results/${dir}_adrc_ryd_gate_rs_shots_s${seed}.json ] && continue
    gate; QW_GPU_GB=12 timeout 7200 runs/capped.sh shots 24 .venv/bin/python paper/alloc/adrc.py $dir --arm adrc_ryd --seed $seed --gate --route_sup --shots 100,1000,10000 >> runs/alloc/adrc.log 2>&1 || echo "CRASH shots $dir s$seed $(date -Is)" >> $S
  done
done
echo "SHOTS_DONE $(date -Is)" >> $S
