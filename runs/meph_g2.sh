#!/bin/bash
# doc 35 §41o G2 on mephiles (moved from the contended Spark): G1 without exploration noise (Adam lr 0.1), then lr 0.01.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/good_start_probe.py --k 2 --steps 1000 --no-noise --tag g2 > runs/l2d/good_start_g2.log 2>&1
$PY paper/l2d/good_start_probe.py --k 2 --steps 1000 --no-noise --lr 0.01 --tag g2lr > runs/l2d/good_start_g2lr.log 2>&1
