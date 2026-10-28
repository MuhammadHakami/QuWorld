#!/bin/bash
# Doc 37 lane 1 (restart after the memguard kill at 00:33): register-neck RB cell first, then the loss/gate ablations.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
ft() { [ -f runs/alloc/ft/rayleigh_benard_${1}_${2}_${3}_s1.json ] && return; mem_gate
  QW_GPU_GB=16 timeout 7200 runs/capped.sh ft1-$1-$2-$3 32 .venv/bin/python paper/alloc/finetune_gated.py rayleigh_benard $1 --loss $2 --head $3 \
    --budget 3600 --val-every 600 --valk 1 --seed 1 >> runs/alloc/ft.log 2>&1 || echo "CRASH ft rayleigh_benard_$1_$2_$3_s1 $(date -Is)" >> $S; }
ft mr30ryd vrmse gate; ft mr30ryd vrmse plain; ft mr30ryd mse gate
echo "FT_LANE1_DONE $(date -Is)" >> $S
