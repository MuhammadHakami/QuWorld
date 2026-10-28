#!/bin/bash
cd ~/Documents/paper/quworld; mkdir -p runs/l2d; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/newton_probe.py --round 3 --run 8 --start cold --barrier 100 --hess 150 --tag n1c > runs/l2d/newton_n1c_r3run8.log 2>&1
