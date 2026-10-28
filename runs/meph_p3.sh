#!/bin/bash
# doc 35 §41a P3 on mephiles: sigmoid vs sin^2 parameterization, paired NAAdamGD + barrier chains, 3,000 steps, R3-1 / 8 / 6.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 1 8 6; do $PY paper/l2d/param_probe.py --round 3 --run $r --steps 3000 > runs/l2d/param_p3_r3run$r.log 2>&1; done
