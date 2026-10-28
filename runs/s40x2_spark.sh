#!/bin/bash
# doc 35 §40x R1 on the Spark (queued behind F1 on the GPU lock): logged Objective calls vs unlogged custom JIT / scan / vmap.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
exec 9>runs/strat/gpu.lock; flock 9
QW_GPU_GB=55 runs/capped.sh l2draw 55 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/raw_bench.py --run 1 --batches 2 4 8 \
  > runs/l2d/raw_bench_spark.log 2>&1
