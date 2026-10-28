#!/bin/bash
# Fusion IPF: PSI-F config seed 1 (as B1_ctrl) on inputs with the organizers' Ip time fix (QW_MAGFIX=X_ipfix), official VALID sample (40).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_MAGFIX=X_ipfix
.venv-vrp/bin/python paper/sophelio/sota_train.py --tag IPF_s1 --score-val 40 \
  --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"}' > runs/strat/fusion/IPF_s1.log 2>&1
