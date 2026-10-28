#!/bin/bash
# doc 35 §41a P3 on the Spark (mephiles runs R3-1 / 8 / 6): sigmoid vs sin^2 parameterization, paired chains, 3,000 steps.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 3 9; do
  QW_GPU_GB=24 runs/capped.sh p3r$r 24 $PY paper/l2d/param_probe.py --round 3 --run $r --steps 3000 > runs/l2d/param_p3_r3run$r.log 2>&1
done
