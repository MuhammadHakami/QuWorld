#!/bin/bash
# doc 35 §41a N1 plateau start (relaunch with an exact start: the 1e-6 clip had turned R = 1 mirrors into 0.999999 -> 0.94).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=30 runs/capped.sh n1p 30 $PY paper/l2d/newton_probe.py --round 3 --run 1 --x0 runs/l2d/hist_submission_v7_e3k_run1.npz --barrier 100 --hess 150 --tag n1p > runs/l2d/newton_n1p_r3run1.log 2>&1
