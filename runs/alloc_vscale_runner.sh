#!/bin/bash
# Doc 35 §19: hybrid scaling with the vanilla (never-register) trunk as the fixed classical part.
# Usage: runs/alloc_vscale_runner.sh <lane> [lanes=2]. One job per lane; MemAvailable >= 22 GB before each launch.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; lane=$1; N=${2:-2}; i=0
jobs() {
  for dir in turbulent_radiative_layer_2D__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337 MHD_64__gate_vanilla_s1337 shear_flow__gate_vanilla_s1337; do
    for seed in 1 2 3; do echo "$dir cls_only $seed 0"; for k in 30 5 10 20; do for arm in adrc_ryd adrc_mlp; do echo "$dir $arm $seed $k"; done; done; done
  done
}
jobs | while read dir arm seed k; do
  i=$((i+1)); [ $((i % N)) -ne $lane ] && continue
  if [ $arm = cls_only ]; then f=runs/alloc/adrc_results/${dir}_cls_only_gate_s${seed}.json; flags="--gate"
  else sfx=$([ $k -ne 30 ] && echo "_k$k"); f=runs/alloc/adrc_results/${dir}_${arm}_gate_rs${sfx}_dyn_s${seed}.json; flags="--gate --route_sup --atoms $k --dyn_tether"; fi
  [ -f $f ] && continue
  until [ -f runs/alloc/adrc_cache/$dir/test.pt ]; do sleep 60; done                        # shear cache builds first
  pgrep -f "adrc.py $dir --arm $arm --seed $seed $flags\$" > /dev/null && continue            # in flight (restarted runner)
  while [ "$(pgrep -fc '^/home/mohhakami/Documents/paper/quworld/.venv/bin/python paper/alloc/adrc.py')" -ge $N ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 22 ]; do sleep 20; done
  QW_GPU_GB=10 timeout 2400 runs/capped.sh vsc$lane 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed $flags >> runs/alloc/vscale_$lane.log 2>&1 || echo "CRASH vscale $dir $arm s$seed k$k $(date -Is)" >> $S
done
echo "VSCALE_LANE_DONE $lane $(date -Is)" >> $S
