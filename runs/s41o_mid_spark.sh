#!/bin/bash
# doc 35 §41o M1: does L-BFGS deepen MID-family designs (dataset loss -0.10 .. +0.25, 4 topologies) the way it deepens good ones
# (-0.21 -> -0.34)? #2 / #5 drop by 0.3-0.5 from +0.06 .. +0.39 between 120 and 180 min on every run.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=30 runs/capped.sh m1 30 $PY paper/l2d/lbfgs_probe.py --start good --band -0.10 0.25 --ntop 4 --iters 300 --tag m1 > runs/l2d/lbfgs_m1_mid.log 2>&1
