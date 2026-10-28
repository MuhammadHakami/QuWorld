#!/bin/bash
# Doc 35 §17: noise-conditioned configuration (FGN-style), K = 30, plus the deterministic seeds 2-3 on TRL2D/MHD_64
# needed for the paired comparison. Usage: runs/alloc_fgn_runner.sh <lane> [lanes=2]; one adrc job at a time per
# runner, and only when MemAvailable >= 22 GB (Sep 25 21:15: VS Code + 4 RouteFinder lanes left 16 GB)
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; lane=$1; N=${2:-2}; i=0
jobs() {   # amendment Sep 26 00:00: vanilla-trunk caches (doc 35 §19); their deterministic K = 30 runs exist for seeds 1-3
  for dir in turbulent_radiative_layer_2D__gate_vanilla_s1337 shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337 MHD_64__gate_vanilla_s1337; do
    for seed in 1 2 3; do for arm in adrc_ryd adrc_mlp; do echo "$dir $arm $seed fgn"; done; done; done
}
jobs | while read dir arm seed mode; do
  i=$((i+1)); [ $((i % N)) -ne $lane ] && continue
  sfx=$([ $mode = fgn ] && echo "_fgn"); f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs_dyn${sfx}_s${seed}.json
  [ -f $f ] && continue
  pgrep -f "adrc.py $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether$([ $mode = fgn ] && echo ' --fgn')\$" > /dev/null && continue
  while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge $N ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 22 ]; do sleep 20; done
  QW_GPU_GB=10 timeout 2400 runs/capped.sh fgn$lane 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup --dyn_tether $([ $mode = fgn ] && echo --fgn) >> runs/alloc/fgn_$lane.log 2>&1 || echo "CRASH fgn $dir $arm s$seed $mode $(date -Is)" >> $S
done
echo "FGN_LANE_DONE $lane $(date -Is)" >> $S
