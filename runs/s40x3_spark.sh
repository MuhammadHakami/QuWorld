#!/bin/bash
# doc 35 §40y rung 1 (O1 optimizer race) on the Spark, queued on the GPU lock: S6 Spark half = R3-1, R3-8 (P), R3-6 (E); 420 s per arm (fixed wall time), 10 arms.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
for rr in "3 1" "3 8" "3 6"; do set -- $rr
  QW_GPU_GB=30 runs/capped.sh l2do1 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/o1_race.py --round $1 --run $2 --seconds 420 \
    >> runs/l2d/o1_spark.log 2>&1
done
