#!/bin/bash
# doc 35 §41p BR1 on mephiles: random binary skeletons + exact laser renormalisation, R3-1 then R3-8 (p1 0.4 / 0.6).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 1 8; do
  for p in 0.4 0.6; do
    $PY paper/l2d/binary_search.py --round 3 --run $r --n 1000 --batch 2 --p1 $p --tag br1p$p >> runs/l2d/binary_br1_r3run$r.log 2>&1
  done
done
