#!/bin/bash
# doc 35 §33b: B3 fixed strong-coupling register (QW_LAM_XY = 0.366), VALID only, after the §31 terminal marker.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; R=runs/alloc/adrc_results_s33; C=runs/alloc/ckpt_s33
until grep -qE "S31_VALID_DONE|S31_INCOMPLETE" $S; do sleep 120; done
for seed in 1 2 3; do for row in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  out=${row}_adrc_ryd_gate_rs_dyn_s${seed}
  [ -f $R/B3lam/$out.json ] && continue
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 26 ]; do sleep 45; done
  env QW_SCREEN=1 QW_LAM_XY=0.366 QW_OUT=$R/B3lam QW_CKPT=$C/B3lam QW_GPU_GB=10 timeout 7200 runs/capped.sh s33b3 20 \
    .venv/bin/python paper/alloc/adrc.py $row --arm adrc_ryd --gate --route_sup --dyn_tether --seed $seed >> runs/alloc/s33.log 2>&1 \
    || { echo "CRASH s33b3 $row s$seed $(date -Is)" >> $S; continue; }
  QW_GPU_GB=10 timeout 3600 runs/capped.sh s33d 20 .venv/bin/python paper/alloc/diag_s33.py $C/B3lam/$out.pt >> runs/alloc/s33.log 2>&1 \
    || echo "CRASH s33b3 diag $row s$seed $(date -Is)" >> $S
done; done; echo "S33B3_DONE $(date -Is)" >> $S
