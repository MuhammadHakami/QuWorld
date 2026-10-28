#!/bin/bash
# doc 35 §41o PSI-W: capacity at full data - PSI-F config (corrected inputs, psi-only, clip 1) with a 4096 x 3 trunk and 150
# epochs, seed 2 (PSI-F seed 2: VALID quick r2_psi 0.999767; PSI-F seeds span 0.999767-0.999859). The 6-seed ensemble did not
# move the composite (bias, not variance) -> test whether the bias is capacity.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
QW_GPU_GB=50 runs/capped.sh psiw 50 .venv/bin/python paper/sophelio/sota_train.py --tag s41o_PSI-W --score-val 0 \
  --cfg '{"k": 64, "width": 4096, "depth": 3, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 150, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 2, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"}' > runs/strat/fusion/s41o_PSI-W.log 2>&1
echo S41O_DONE >> runs/strat/fusion/s41o_PSI-W.log
