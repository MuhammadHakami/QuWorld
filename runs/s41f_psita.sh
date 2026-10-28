#!/bin/bash
# doc 35 §41f: PSI-T (Thomson replaced) is worse than PSI-F at full data on 2/2 paired seeds; one seed of the T41 variant
# (fixed-coordinate Thomson ADDED to the old block) vs PSI-F seed 2 (VALID quick 0.999767).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "thz": true, "seed": 2'
QW_GPU_GB=40 runs/capped.sh psita 60 .venv/bin/python paper/sophelio/sota_train.py --tag s41f_PSI-TA --score-val 0 --cfg "{$CFG}" > runs/strat/fusion/s41f_PSI-TA.log 2>&1
