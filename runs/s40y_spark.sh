#!/bin/bash
# doc 35 §40y rung 1 (O1 race, step budget: 380 gradient-equivalent steps per arm, 10 arms) on the Spark: the three S6
# Spark topologies R3-1, R3-8, R3-6 run concurrently (user: parallel hypotheses where memory allows; ~25 GB each).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
for r in 1 8 6; do
  QW_GPU_GB=25 runs/capped.sh l2do1r$r 25 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/o1_race.py --round 3 --run $r --units 380 \
    > runs/l2d/o1_spark_r3run$r.log 2>&1 &
  sleep 90
done
wait
