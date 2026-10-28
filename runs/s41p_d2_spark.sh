#!/bin/bash
# doc 35 §41p D2: which update rule keeps descending from our v7a plateau design (7 rules in 7 vmap'd lanes), R3-1 then R3-8.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 1 8; do
  QW_GPU_GB=30 runs/capped.sh d2r$r 30 $PY paper/l2d/descent_race.py --round 3 --run $r --steps 2000 > runs/l2d/descent_d2_r3run$r.log 2>&1
done
