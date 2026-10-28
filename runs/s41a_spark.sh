#!/bin/bash
# doc 35 §41a N1: exact-Hessian trust-region Newton (one Hessian + up to 4 value trials per iteration, #1's evaluation
# profile) on the smooth power-barrier surrogate, Round 3 topology 1 (our official 0.416, #1 -0.618), two starts side by
# side: cold random (does second-order descent land below the 0.42 plateau, as #1's 15-min checkpoints suggest?) and our
# v7a plateau design (0.457; can second-order steps leave the plateau?). 150 Hessians each; read-out per Hessian.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=30 runs/capped.sh n1c 30 $PY paper/l2d/newton_probe.py --round 3 --run 1 --start cold --barrier 100 --hess 150 --tag n1c > runs/l2d/newton_n1c_r3run1.log 2>&1 &
sleep 240
QW_GPU_GB=30 runs/capped.sh n1p 30 $PY paper/l2d/newton_probe.py --round 3 --run 1 --x0 runs/l2d/hist_submission_v7_e3k_run1.npz --barrier 100 --hess 150 --tag n1p > runs/l2d/newton_n1p_r3run1.log 2>&1 &
wait
