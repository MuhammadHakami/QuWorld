#!/bin/bash
# mephiles queue: v9b smoke (2 lanes, GPU now free), then PSI-E seeds 1 / 3 (200 epochs; PSI-F seeds 1 / 3: 0.999859 / 0.999834).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --round 3 --run 1 --sub paper/l2d/submission_v9 --evals 200 \
  --kw '{"lanes": 2, "k": 4, "k_wide": 4, "cold_after": 20, "cold_len": 40}' --tag v9bsmoke > runs/l2d/sub_v9bsmoke.log 2>&1
for s in 1 3; do
  .venv-vrp/bin/python paper/sophelio/sota_train.py --tag s41o_PSI-E-s$s --score-val 0 \
    --cfg "{\"k\": 64, \"width\": 2048, \"depth\": 2, \"drop\": 0.1, \"lr\": 0.0025322139193052723, \"batch\": 4096, \"epochs\": 200, \"w_scal\": 0.0, \"gamma\": 0.3152355297039971, \"lags\": \"sym\", \"thomson\": true, \"quantum\": \"none\", \"k_lo\": 32, \"q_layers\": 4, \"seed\": $s, \"edge_w\": 2.0, \"clip\": 1.0, \"fast\": true, \"inputs\": \"magfix\"}" > runs/strat/fusion/s41o_PSI-E-s$s.log 2>&1
  echo S41O_DONE >> runs/strat/fusion/s41o_PSI-E-s$s.log
done
