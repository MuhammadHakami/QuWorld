#!/bin/bash
# doc 35 §40y O2 evolution race on the Spark after the three O1 races finish: R3-1, R3-8, R3-6 concurrently (env-step budget).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
while pgrep -f "o1_race[.]py" > /dev/null; do sleep 60; done
for r in 1 8 6; do
  QW_GPU_GB=25 runs/capped.sh l2do2r$r 25 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/evo_race.py --round 3 --run $r --units 380 \
    > runs/l2d/o2_spark_r3run$r.log 2>&1 &
  sleep 90
done
wait
