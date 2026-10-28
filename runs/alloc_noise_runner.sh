#!/bin/bash
# Doc 35 §12: eval-only calibration noise on the route-supervised register (retrained as in §9, same seeds).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for dir in rayleigh_benard__gatest_mr30ryd_s1337 shear_flow__gate_mr30ryd_s1337; do
  for seed in 1 2 3; do
    [ -f runs/alloc/adrc_results/${dir}_adrc_ryd_gate_rs_noise_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 2400 runs/capped.sh noise 24 .venv/bin/python paper/alloc/adrc.py $dir --arm adrc_ryd --seed $seed --gate --route_sup --noise >> runs/alloc/adrc.log 2>&1 || echo "CRASH noise $dir s$seed $(date -Is)" >> $S
  done
done
echo "NOISE_DONE $(date -Is)" >> $S
