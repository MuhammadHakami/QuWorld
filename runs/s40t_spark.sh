#!/bin/bash
# doc 35 §40t (extension): cold NAAdamGD to 10,000 evaluations on topology 6 (CUDA, Spark), alone under the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/warm_vs_cold.py --run 6 --evals 10000 --arms cold --tag _long \
  > runs/l2d/wvc_long_run6.log 2>&1
