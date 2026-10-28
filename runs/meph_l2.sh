#!/bin/bash
# doc 35 §41j L2 on mephiles: L-BFGS polish of our v7a plateau designs (R3-1 / 8 / 6; v7a 3k: 0.457 / 0.450 / 0.632; official v6
# finals 0.416 / 0.411 / 0.157), 400 iterations, stock dense FP64 solver (the block solver is not robust on high-finesse systems); then G1's good designs again.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/lbfgs_probe.py --start hist --round 3 --runs 1 8 6 --iters 400 --tag l2 > runs/l2d/lbfgs_l2_hist.log 2>&1
$PY paper/l2d/lbfgs_probe.py --start good --iters 400 --tag l1b > runs/l2d/lbfgs_l1b_good.log 2>&1
