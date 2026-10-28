#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
flock runs/strat/gpu.lock env QW_GPU_GB=24 runs/capped.sh s38i 40 .venv/bin/python paper/sophelio/sota_train.py --tag s38i_PSI-F-s1 --score-val 0 \
  --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"}' > runs/strat/fusion/s38i_PSI-F-s1.log 2>&1
echo S38I_DONE >> runs/strat/fusion/s38i_PSI-F-s1.log
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
flock runs/strat/gpu.lock env QW_GPU_GB=24 runs/capped.sh s38i 40 .venv/bin/python paper/sophelio/sota_train.py --tag s38i_PSI-F-s3 --score-val 0 \
  --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 3, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"}' > runs/strat/fusion/s38i_PSI-F-s3.log 2>&1
echo S38I_DONE >> runs/strat/fusion/s38i_PSI-F-s3.log
