#!/bin/bash
# doc 35 §41p v9 (7 batched lanes) at matched H100 wall time: 15 min on the H100 = ~1,400 calls of 7 = 9,800 evaluations
# (v7a: ~3,600 single evaluations). R3-8 then R3-6 on merphistophline (ROCm, one JAX process at a time).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
for r in 8 6; do
  $PY paper/l2d/run_submission.py --round 3 --run $r --sub paper/l2d/submission_v9 --evals 9800 --tag submission_v9_e9k8 > runs/l2d/sub_v9_e9k8_r3run$r.log 2>&1
done
