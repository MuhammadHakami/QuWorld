#!/bin/bash
# doc 35 §41o PSI-E on mephiles (3080 Ti, .venv-vrp torch 2.14 cu130): PSI-F config, seed 2, 200 epochs (vs 90) - isolates the
# epoch count from PSI-W's capacity change (4096 x 3, 150 epochs; Spark). Reference: PSI-F seed 2 VALID quick 0.999767.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1
.venv-vrp/bin/python paper/sophelio/sota_train.py --tag s41o_PSI-E --score-val 0 \
  --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 200, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 2, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"}' > runs/strat/fusion/s41o_PSI-E.log 2>&1
echo S41O_DONE >> runs/strat/fusion/s41o_PSI-E.log
