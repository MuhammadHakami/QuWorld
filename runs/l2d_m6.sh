#!/bin/bash
# mephiles (CUDA): §40s v5 (xcell) runs 3 and 6.
cd ~/Documents/paper/quworld; export XLA_PYTHON_CLIENT_PREALLOCATE=false PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 3 6; do $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v5 --kw '{"hop_mode": "xcell", "patience": 500, "seg": 150, "lr_hop": 0.01, "pool_every": 30}' > runs/l2d/sub_v5x_run$r.log 2>&1; done
echo "L2D_M6_DONE $(date -Is)" >> runs/l2d/m6.log
