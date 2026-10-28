#!/bin/bash
# Doc 35 §20: K = 50 on the vanilla-trunk caches (shear_flow, rayleigh_benard), register and MLP, seeds 1-3; one lane.
# Sep 26 03:25: restarted with a 4-job cap (adrc jobs use <1 GB GPU, ~2 GB RSS) so the registered RB test is not starved.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for dir in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337; do for seed in 1 2 3; do for arm in adrc_ryd adrc_mlp; do
  f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs_k50_dyn_s${seed}.json; [ -f $f ] && continue
  while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge 4 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 18 ]; do sleep 20; done
  QW_GPU_GB=10 timeout 3600 runs/capped.sh k50 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --atoms 50 --dyn_tether >> runs/alloc/k50.log 2>&1 || echo "CRASH k50 $dir $arm s$seed $(date -Is)" >> $S
done; done; done
echo "K50_DONE $(date -Is)" >> $S
