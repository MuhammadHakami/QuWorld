#!/bin/bash
# doc 35 §41d N2 on mephiles: interior-point trust-region Newton (scipy trust-constr, exact JAX Hessians, bounded space),
# R3-1 from the v6 warm start (best of the 64-design transfer pool), 200 iterations.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/ipn_probe.py --round 3 --run 1 --iters 200 --tag n2 > runs/l2d/ipn_n2_r3run1.log 2>&1
