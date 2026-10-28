#!/bin/bash
# §43f screen (registered before any result): a bigger psi trunk. §43e shows head / psi accuracy is the whole remaining gap and #1's psi has
# ~1/3 of our error; trunks were never scaled past 2048 x 2 (only epochs / loss). PQ2W_s1 = PQ2 recipe (k 64, qphys2, TRAIN only, seed 1) at
# width 4096, depth 4, 60 epochs. Gate: VALID quick r2_psi vs PQ2's 0.999865 (same recipe / seed at 2048 x 2); iff > 0.99990 -> the VT chain
# read-out (raw + VB vs PQ2) is registered next.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43f.log
CFG='"k": 64, "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 60, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "qphys": 2'
[ -f runs/sophelio_sota/PQ2W.pt ] || QW_GPU_GB=30 runs/capped.sh s43f 40 .venv/bin/python paper/sophelio/sota_train.py --tag PQ2W --score-val 40 \
  --cfg "{$CFG}" >> $LOG 2>&1 || { echo "CRASH PQ2W $(date -Is)" >> $LOG; exit 1; }
echo "PQ2W_TRAINED $(date -Is)" >> $LOG
