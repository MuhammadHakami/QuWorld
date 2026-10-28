#!/bin/bash
# Fusion B1c on mephiles: PSI-F config seed 1, 90 epochs, control vs LCFS-contour loss (lcfs_w 1 / 5); official VALID-sample
# composite on 40 shots (score_val) - the boundary/shape terms carry our gap.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1
for arm in "B1_ctrl|0.0" "B1_lc1|1.0" "B1_lc5|5.0"; do
  name=${arm%%|*}; w=${arm#*|}
  .venv-vrp/bin/python paper/sophelio/sota_train.py --tag $name --score-val 40 \
    --cfg "{\"k\": 64, \"width\": 2048, \"depth\": 2, \"drop\": 0.1, \"lr\": 0.0025322139193052723, \"batch\": 4096, \"epochs\": 90, \"w_scal\": 0.0, \"gamma\": 0.3152355297039971, \"lags\": \"sym\", \"thomson\": true, \"quantum\": \"none\", \"k_lo\": 32, \"q_layers\": 4, \"seed\": 1, \"edge_w\": 2.0, \"clip\": 1.0, \"fast\": true, \"inputs\": \"magfix\", \"lcfs_w\": $w}" > runs/strat/fusion/$name.log 2>&1
done
