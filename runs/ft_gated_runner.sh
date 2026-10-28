#!/bin/bash
# Doc 37: end-to-end gated fine-tuning factorial on rayleigh_benard (stratified VALID selection), 2 concurrent lanes.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
ft() { local tag=$1_$2_$3_$4_s1
  [ -f runs/alloc/ft/rayleigh_benard_${2}_${3}_${4}_s1.json ] && return
  QW_GPU_GB=16 timeout 7200 runs/capped.sh ft-$2-$3-$4 32 .venv/bin/python paper/alloc/finetune_gated.py $1 $2 --loss $3 --head $4 \
    --budget 3600 --val-every 600 --valk 1 --seed 1 >> runs/alloc/ft.log 2>&1 || echo "CRASH ft $tag $(date -Is)" >> $S; }
( ft rayleigh_benard mr30ryd vrmse gate; ft rayleigh_benard mr30ryd vrmse plain; ft rayleigh_benard mr30ryd mse gate ) &
( ft rayleigh_benard vanilla vrmse gate; ft rayleigh_benard mlp30 vrmse gate ) &
wait; echo "FT_RB_DONE $(date -Is)" >> $S
