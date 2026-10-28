#!/bin/bash
# doc 35 §41n F1: binary flip neighbourhoods (tunings 0/90, reflectivities 0/1, lengths 1 m/4 km) + laser renormalisation +
# L-BFGS repair, from our v7a plateau designs R3-1 / R3-8 / R3-6.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 1 8 6; do
  QW_GPU_GB=30 runs/capped.sh f1r$r 30 $PY paper/l2d/flip_probe.py --round 3 --run $r --rounds 4 --batch 4 > runs/l2d/flip_f1_r3run$r.log 2>&1
done
