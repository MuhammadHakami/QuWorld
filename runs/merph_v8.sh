#!/bin/bash
# doc 35 §41m on merphistophline (ROCm venv, one JAX process at a time): v8 (v7a + closing L-BFGS from 80% of the budget)
# end to end, 3,000 evaluations, R3-1 / 8 / 6; polish gain = v8 at 3,000 vs v8 at 2,400, against v7a's 2,400 -> 3,000 gain.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
for r in 1 8 6; do
  $PY paper/l2d/run_submission.py --round 3 --run $r --sub paper/l2d/submission_v8 --evals 3000 --tag submission_v8_e3k > runs/l2d/sub_v8_e3k_r3run$r.log 2>&1
done
echo V8_DONE >> runs/l2d/sub_v8_e3k_r3run6.log
