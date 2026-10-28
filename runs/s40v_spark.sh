#!/bin/bash
# doc 35 §40v: v6 (submission 420 defaults) faithful to the first 30,000 evaluations of the official run, topology 6 (Spark, alone).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --run 6 --evals 30000 --tag v6f \
  > runs/l2d/sub_v6f_run6.log 2>&1
