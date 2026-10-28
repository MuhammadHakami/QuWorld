#!/bin/bash
# doc 35 §26b: capacity (adrc_mlpw, 297) and graph-aware (adrc_gnnm, 239) controls for the §26 shear K=100 pass; seeds 4-8.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
cell() { local dir=$1 arm=$2 seed=$3
  [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_rs_k100_dyn_s${seed}.json ] && return 0
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 28 ]; do sleep 60; done
  QW_GPU_GB=14 timeout 7200 runs/capped.sh s26b 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate --route_sup \
    --atoms 100 --dyn_tether >> runs/alloc/s26.log 2>&1 || echo "CRASH s26b $dir $arm s$seed $(date -Is)" >> $S; }
for seed in 4 5 6 7 8; do for arm in adrc_gnnm adrc_mlpw; do cell shear_flow__gate_vanilla_s1337 $arm $seed; done; done
echo "S26B_SHEAR_DONE $(date -Is)" >> $S
until [ -f runs/alloc/adrc_results/rayleigh_benard__gatest_vanilla_s1337_adrc_ryd_gate_rs_k100_dyn_s8.json ]; do sleep 300; done   # RB conditional on its §26 K=100 cell passing
if .venv/bin/python - <<'PY'
import json, glob, sys
from scipy.stats import ttest_rel
d = "runs/alloc/adrc_results/rayleigh_benard__gatest_vanilla_s1337_adrc_{}_gate_rs_k100_dyn_s{}.json"
p = [(json.load(open(d.format("ryd", s)))["test"], json.load(open(d.format("mlp", s)))["test"]) for s in range(4, 9)]
r, m = zip(*p); ok = sum(a < b for a, b in p) >= 4 and ttest_rel(r, m, alternative="less").pvalue < 0.05
sys.exit(0 if ok else 1)
PY
then for seed in 4 5 6 7 8; do for arm in adrc_gnnm adrc_mlpw; do cell rayleigh_benard__gatest_vanilla_s1337 $arm $seed; done; done; fi
echo "S26B_DONE $(date -Is)" >> $S
