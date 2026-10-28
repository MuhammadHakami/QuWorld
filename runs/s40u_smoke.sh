#!/bin/bash
# doc 35 §40u smoke: warm v3b for 1,000 evaluations, then cold NAAdamGD runs of 300 (code path check), topology 1, Spark alone.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --run 1 --evals 1600 --tag v6smoke \
  --kw '{"cold_after": 1000, "cold_len": 300}' > runs/l2d/sub_v6smoke_run1.log 2>&1
