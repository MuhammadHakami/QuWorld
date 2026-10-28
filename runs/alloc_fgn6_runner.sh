#!/bin/bash
# Doc 35 §17 power extension (registered 03:40 Sep 26, after seeds 1-3): FGN seeds 4-6, TRL2D + shear_flow, both cores.
# Sep 26 04:13: restarted serial (one adrc job at a time) so the doc 38 TD replication keeps the GPU.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for seed in 4 5 6; do for dir in turbulent_radiative_layer_2D__gate_vanilla_s1337 shear_flow__gate_vanilla_s1337; do for arm in adrc_ryd adrc_mlp; do
  f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs_dyn_fgn_s${seed}.json
  for try in 1 2 3; do
    [ -f $f ] && break
    while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge 1 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 18 ]; do sleep 20; done
    QW_GPU_GB=10 timeout 2400 runs/capped.sh fgn6 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether --fgn >> runs/alloc/fgn6.log 2>&1 || echo "CRASH fgn6 $dir $arm s$seed try$try $(date -Is)" >> $S
  done
done; done; done
echo "FGN6_DONE $(date -Is)" >> $S
