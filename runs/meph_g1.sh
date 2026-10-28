#!/bin/bash
# doc 35 §41g G1 on mephiles: incumbent chain started at good dataset designs (< -0.2) on their own topologies, 2,000 steps.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/good_start_probe.py --k 3 --steps 2000 > runs/l2d/good_start_g1.log 2>&1
