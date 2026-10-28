#!/bin/bash
# Control for QFI: same QF config on the ORIGINAL magfix inputs (machine / code-version check; s38g_QF seed 2 = q95 0.99602).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_MAGFIX=X_magfix
for s in 2; do
  .venv-vrp/bin/python paper/sophelio/sota_train.py --tag QFC_s$s --score-val 0 \
    --cfg "{\"k\": 64, \"width\": 2048, \"depth\": 2, \"drop\": 0.1, \"lr\": 0.0025322139193052723, \"batch\": 4096, \"epochs\": 30, \"w_scal\": 1.5123263659215784, \"gamma\": 0.3152355297039971, \"lags\": \"sym\", \"thomson\": true, \"quantum\": \"none\", \"k_lo\": 32, \"q_layers\": 4, \"edge_w\": 2.0, \"fast\": true, \"inputs\": \"magfix\", \"seed\": $s}" > runs/strat/fusion/QFC_s$s.log 2>&1
done
