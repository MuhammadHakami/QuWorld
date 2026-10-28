#!/bin/bash
# doc 35 §41p single-chain strategy tests on merphistophline (ROCm; batched runs there hit NaN resets), R3-1, 6,000 evaluations each:
#   w_lr2  warm (best pool design), Adam 0.01, no noise, no pool restarts
#   c_lr2  cold random start, Adam 0.01, no noise, one long run
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
$PY paper/l2d/run_submission.py --round 3 --run 1 --sub paper/l2d/submission_v9 --evals 6000 --tag single_wlr2_e6k \
  --kw '{"lanes": 1, "learning_rate": 0.01, "noise_start": 0.0, "patience": 100000, "step_patience": 100000, "cold_after": 1000000}' > runs/l2d/sub_single_wlr2_e6k_r3run1.log 2>&1
$PY paper/l2d/run_submission.py --round 3 --run 1 --sub paper/l2d/submission_v9 --evals 6000 --tag single_clr2_e6k \
  --kw '{"lanes": 1, "learning_rate": 0.01, "k": 1, "k_wide": 1, "cold_after": 1, "cold_s0": 0.0, "cold_len": 1000000, "cold_settle": []}' > runs/l2d/sub_single_clr2_e6k_r3run1.log 2>&1
