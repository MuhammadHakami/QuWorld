#!/bin/bash
# v9 (fixed NaN guard) on CUDA with 2 lanes (3 OOM on the 3080 Ti), R3-8, 5,600 evaluations (~15 H100-min at batch 2): checks the
# merphistophline ROCm result (1.055, NaN resets) against CUDA.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --round 3 --run 8 --sub paper/l2d/submission_v9 --evals 5600 \
  --kw '{"lanes": 2}' --tag submission_v9_L2_e5k6 > runs/l2d/sub_v9_L2_e5k6_r3run8.log 2>&1
