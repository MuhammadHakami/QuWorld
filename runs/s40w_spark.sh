#!/bin/bash
# doc 35 §40w H1 on the Spark (alone under the GPU lock): trust-region Newton, Round 2 topology 1, warm start, 40 Hessians.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/newton_probe.py --run 1 --hess 40 --start warm --tag h1 \
  > runs/l2d/newton_h1_run1.log 2>&1
