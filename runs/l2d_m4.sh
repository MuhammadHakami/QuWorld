#!/bin/bash
# mephiles (L2D only, CUDA): §40o - v3b on run 1 for 10,000 evaluations (submission defaults), trace saved.
cd ~/Documents/paper/quworld; export XLA_PYTHON_CLIENT_PREALLOCATE=false PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "[v]alidate_submission.py|[r]un_submission.py" > /dev/null; do sleep 30; done
$PY paper/l2d/run_submission.py --run 1 --evals 10000 --tag v3b_long > runs/l2d/sub_v3b_long_run1.log 2>&1
echo "L2D_M4_DONE $(date -Is)" >> runs/l2d/m4.log
