#!/bin/bash
# mephiles (L2D only; routing work done): §40l-b diagnostic - v2b and v1 on run 1, 1,200 evaluations each, per-evaluation
# loss/feasibility traces (runs/l2d/hist_*.npz) to see why v2b froze at 1.043 from evaluation 500.
cd ~/Documents/paper/quworld; export XLA_PYTHON_CLIENT_PREALLOCATE=false PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/run_submission.py --run 1 --evals 1200 --tag diag_v2b --kw '{"lr_steps": [0.01, 0.003], "step_patience": 500}' > runs/l2d/sub_diag_v2b_run1.log 2>&1
$PY paper/l2d/run_submission.py --run 1 --evals 1200 --tag diag_v1 > runs/l2d/sub_diag_v1_run1.log 2>&1
echo "L2D_M_DIAG_DONE $(date -Is)" >> runs/l2d/m_diag.log
