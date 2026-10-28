#!/bin/bash
# doc 35 §41p: single-chain v9b on R3-6 with SHORT cold runs (1,500 steps, settle at 60 / 85%), 16k evaluations (vs 4,000-step runs: 0.500).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
baseline/Learn2Design-2026/.venv-rocm/bin/python paper/l2d/run_submission.py --round 3 --run 6 --sub paper/l2d/submission_v9 --evals 16000 \
  --kw '{"lanes": 1, "cold_after": 3000, "cold_len": 1500}' --tag single_v9b_c15_e16k > runs/l2d/sub_single_v9b_c15_e16k_r3run6.log 2>&1
