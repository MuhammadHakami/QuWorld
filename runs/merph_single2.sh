#!/bin/bash
# doc 35 §41p single-chain v9b on R3-6 (PoolHop reached 0.157 there): short warm phase (3k), cold runs of 4k steps ending in the
# lr 0.01 / 0.003 settle, 16k evaluations - do settled cold runs reach the mid/good family on a topology where it exists?
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
baseline/Learn2Design-2026/.venv-rocm/bin/python paper/l2d/run_submission.py --round 3 --run 6 --sub paper/l2d/submission_v9 --evals 16000 \
  --kw '{"lanes": 1, "cold_after": 3000, "cold_len": 4000}' --tag single_v9b_e16k > runs/l2d/sub_single_v9b_e16k_r3run6.log 2>&1
