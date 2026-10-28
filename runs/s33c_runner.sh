#!/bin/bash
# doc 35 §33c: retrained interaction-free register (QW_TAU0), VALID only, after §33b B3.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; R=runs/alloc/adrc_results_s33; C=runs/alloc/ckpt_s33
until grep -q S33B3_DONE $S; do sleep 120; done
for seed in 1 2 3; do for row in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  out=${row}_adrc_ryd_gate_rs_dyn_s${seed}
  [ -f $R/C0tau0/$out.json ] && continue
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 26 ]; do sleep 45; done
  env QW_SCREEN=1 QW_TAU0=1 QW_OUT=$R/C0tau0 QW_CKPT=$C/C0tau0 QW_GPU_GB=10 timeout 7200 runs/capped.sh s33c 20 \
    .venv/bin/python paper/alloc/adrc.py $row --arm adrc_ryd --gate --route_sup --dyn_tether --seed $seed >> runs/alloc/s33.log 2>&1 \
    || { echo "CRASH s33c $row s$seed $(date -Is)" >> $S; continue; }
  QW_GPU_GB=10 timeout 3600 runs/capped.sh s33d 20 .venv/bin/python paper/alloc/diag_s33.py $C/C0tau0/$out.pt >> runs/alloc/s33.log 2>&1 \
    || echo "CRASH s33c diag $row s$seed $(date -Is)" >> $S
done; done; echo "S33C_DONE $(date -Is)" >> $S
