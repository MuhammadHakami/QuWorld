#!/bin/bash
# Fusion IPF: PSI-F config seeds 2 / 3 on X_ipfix (psi ensemble for the HJl9-on-ipfix rebuild), quick VALID only.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_MAGFIX=X_ipfix
for s in 2 3; do
  .venv-vrp/bin/python paper/sophelio/sota_train.py --tag IPF_s$s --score-val 0 \
    --cfg "{\"k\": 64, \"width\": 2048, \"depth\": 2, \"drop\": 0.1, \"lr\": 0.0025322139193052723, \"batch\": 4096, \"epochs\": 90, \"w_scal\": 0.0, \"gamma\": 0.3152355297039971, \"lags\": \"sym\", \"thomson\": true, \"quantum\": \"none\", \"k_lo\": 32, \"q_layers\": 4, \"seed\": $s, \"edge_w\": 2.0, \"clip\": 1.0, \"fast\": true, \"inputs\": \"magfix\"}" > runs/strat/fusion/IPF_s$s.log 2>&1
done
