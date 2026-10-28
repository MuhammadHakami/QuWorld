#!/bin/bash
# doc 35 §33: VALID-only redesign (QW_SCREEN: TEST never scored). Stage A diagnose, B1 tether estimator, B2 objective 2x2.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; R=runs/alloc/adrc_results_s33; C=runs/alloc/ckpt_s33
mem() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 26 ]; do sleep 45; done; }
one() {  # cell seed row out_suffix envs... -- arm flags...
  local cell=$1 seed=$2 row=$3 out=$4; shift 4; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -f $R/$cell/${row}_${out}_s${seed}.json ] && return 0
  mem
  env QW_SCREEN=1 QW_OUT=$R/$cell QW_CKPT=$C/$cell "${envs[@]}" QW_GPU_GB=10 timeout 7200 runs/capped.sh s33 20 \
    .venv/bin/python paper/alloc/adrc.py $row "$@" --seed $seed >> runs/alloc/s33.log 2>&1 || { echo "CRASH s33 $cell $row $* s$seed $(date -Is)" >> $S; return 1; }
  [ -f $C/$cell/${row}_${out}_s${seed}.pt ] && QW_GPU_GB=10 timeout 3600 runs/capped.sh s33d 20 .venv/bin/python paper/alloc/diag_s33.py \
    $C/$cell/${row}_${out}_s${seed}.pt >> runs/alloc/s33.log 2>&1 || echo "CRASH s33 diag $cell $row s$seed $(date -Is)" >> $S
}
ROWS="shear_flow__gate_vanilla_s1337 rayleigh_benard__gatest_vanilla_s1337"
arms() {  # cell seed row envs...
  local cell=$1 seed=$2 row=$3; shift 3
  one $cell $seed $row adrc_ryd_gate_rs_dyn "$@" -- --arm adrc_ryd --gate --route_sup --dyn_tether
  one $cell $seed $row adrc_mlp_gate_rs_dyn "$@" -- --arm adrc_mlp --gate --route_sup --dyn_tether
}
for seed in 1 2 3; do for row in $ROWS; do                                                   # stage A (+ cls_only)
  one A $seed $row cls_only_gate -- --arm cls_only --gate
  arms A $seed $row
done; done
for seed in 1 2 3; do for row in $ROWS; do                                                   # B1 tether estimator
  arms B1det $seed $row QW_TETHER_MODE=det
  arms B1off $seed $row QW_TETHER_MODE=off
done; done
for seed in 1 2; do for row in $ROWS; do                                                     # B2 objective 2x2
  arms B2loss $seed $row QW_LOSS=vrmse
  arms B2tgt $seed $row QW_ROUTE_TGT=vrmse
  arms B2both $seed $row QW_LOSS=vrmse QW_ROUTE_TGT=vrmse
done; done
echo "S33_DONE $(date -Is)" >> $S
