#!/bin/bash
# doc 35 §41p v9 (7 batched lanes) at matched H100 wall time (15 min = 9,800 evaluations), R3-1 on the Spark.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=50 runs/capped.sh v9r1 50 $PY paper/l2d/run_submission.py --round 3 --run 1 --sub paper/l2d/submission_v9 --evals 9800 --tag submission_v9_e9k8 > runs/l2d/sub_v9_e9k8_r3run1.log 2>&1
