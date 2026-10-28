#!/bin/bash
# doc 35 §33a: retrained no-core arm (VALID only), after §33 stage A; never runs alongside a §33 job.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; R=runs/alloc/adrc_results_s33; C=runs/alloc/ckpt_s33
until [ "$(ls $R/A/*_s[123].json 2>/dev/null | wc -l)" -ge 18 ]; do sleep 120; done
for seed in 1 2 3; do for row in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do
  [ -f $R/Anocore/${row}_adrc_ryd_gate_rs_dyn_s${seed}.json ] && continue
  while pgrep -f "capped.sh s33 " >/dev/null || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 26 ]; do sleep 45; done
  env QW_SCREEN=1 QW_NO_CORE=1 QW_OUT=$R/Anocore QW_CKPT=$C/Anocore QW_GPU_GB=10 timeout 7200 runs/capped.sh s33a 20 \
    .venv/bin/python paper/alloc/adrc.py $row --arm adrc_ryd --gate --route_sup --dyn_tether --seed $seed >> runs/alloc/s33.log 2>&1 \
    || echo "CRASH s33a $row s$seed $(date -Is)" >> $S
done; done; echo "S33A_DONE $(date -Is)" >> $S
