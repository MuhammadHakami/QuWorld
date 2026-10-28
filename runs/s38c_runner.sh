#!/bin/bash
# doc 35 §38c: full-data single-seed specialists, one at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
S=runs/alloc/status; L=runs/sophelio_sota/s38c.log
BASE='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 30, "w_scal": 1.5123263659215784, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 2, "edge_w": 2.0'
for arm in PSI-A QB-A; do
  [ -f runs/sophelio_sota/s38c_$arm.json ] && continue
  case $arm in PSI-A) X=', "epochs": 90, "clip": 1.0, "w_scal": 0.0';; QB-A) X=', "depth": 4, "epochs": 90';; esac
  flock runs/strat/gpu.lock env QW_GPU_GB=40 timeout 7200 runs/capped.sh s38c 60 .venv/bin/python paper/sophelio/sota_train.py \
    --tag s38c_$arm --cfg "{$BASE$X}" --score-val 0 >> $L 2>&1 || echo "CRASH s38c $arm $(date -Is)" >> $S
done
echo "S38C_DONE $(date -Is)" >> $S
