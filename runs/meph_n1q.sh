#!/bin/bash
# doc 35 §41a N1 continued on mephiles: trust-region Newton (barrier surrogate) from the Spark n1p iterate after 14 Hessians
# (R3-1, best 0.4519), radius carried over (0.0625), up to 300 more Hessians (~30 s each here).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/newton_probe.py --round 3 --run 1 --x0 runs/l2d/hist_n1p_iter_run1.npz --delta0 0.0625 --barrier 100 --hess 300 --tag n1q > runs/l2d/newton_n1q_r3run1.log 2>&1
