#!/bin/bash
# doc 35 §41p: Spark ALONE (no other GPU job): v9 (fixed NaN guard) R3-1 at matched H100 time (9,800 evaluations, 7 lanes),
# then D3 lane race R3-1 (6,000 steps).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=60 runs/capped.sh v9r1 60 $PY paper/l2d/run_submission.py --round 3 --run 1 --sub paper/l2d/submission_v9 --evals 9800 --tag submission_v9_e9k8 > runs/l2d/sub_v9_e9k8_r3run1.log 2>&1
QW_GPU_GB=60 runs/capped.sh d3r1 60 $PY paper/l2d/lane_race.py --round 3 --run 1 --steps 6000 > runs/l2d/lanes_d3_r3run1.log 2>&1
