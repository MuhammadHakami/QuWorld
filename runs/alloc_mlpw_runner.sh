#!/bin/bash
# Doc 35 §17 wider-MLP control (registered 05:08 Sep 26): adrc_mlpw (nearest width + 1 hidden unit), FGN, seeds 1-6,
# shear_flow then TRL2D; serial (one adrc job at a time) so the doc 38 TD replication keeps the GPU.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for dir in shear_flow__gate_vanilla_s1337 turbulent_radiative_layer_2D__gate_vanilla_s1337; do for seed in 1 2 3 4 5 6; do
  f=runs/alloc/adrc_results/${dir}_adrc_mlpw_gate_rs_dyn_fgn_s${seed}.json
  for try in 1 2 3; do
    [ -f $f ] && break
    while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge 1 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 18 ]; do sleep 20; done
    QW_GPU_GB=10 timeout 2400 runs/capped.sh mlpw 16 .venv/bin/python paper/alloc/adrc.py $dir --arm adrc_mlpw --seed $seed --gate --route_sup --dyn_tether --fgn >> runs/alloc/mlpw.log 2>&1 || echo "CRASH mlpw $dir s$seed try$try $(date -Is)" >> $S
  done
done; done
echo "MLPW_DONE $(date -Is)" >> $S
