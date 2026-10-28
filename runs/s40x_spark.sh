#!/bin/bash
# doc 35 §40x F1 on the Spark (alone under the GPU lock): bang-bang flip sweeps after Adam, Round 2 topology 1, warm start.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/flip_probe.py --run 1 --adam 1000 --sweeps 4 --refine 150 --tag f1 \
  > runs/l2d/flip_f1_r2run1.log 2>&1
