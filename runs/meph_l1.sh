#!/bin/bash
# doc 35 §41j L1 on mephiles: L-BFGS refinement (curvature-scaled) from good dataset designs, then from the R3-1 transfer pools.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "[g]ood_start_probe" > /dev/null; do sleep 30; done
$PY paper/l2d/lbfgs_probe.py --start good --iters 400 --fast > runs/l2d/lbfgs_l1_good.log 2>&1
$PY paper/l2d/lbfgs_probe.py --start pool --round 3 --run 1 --k 6 --iters 400 --fast > runs/l2d/lbfgs_l1_pool_r3run1.log 2>&1
