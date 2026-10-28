#!/bin/bash
# Spark (alone, GPU lock): §40s v5 (xcell) runs 1 and 4.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
for r in 1 4; do QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --run $r --evals 3000 --tag v5 \
  --kw '{"hop_mode": "xcell", "patience": 500, "seg": 150, "lr_hop": 0.01, "pool_every": 30}' > runs/l2d/sub_v5x_run$r.log 2>&1; done
echo "L2D_SPARK_Q7_DONE $(date -Is)" >> runs/l2d/spark_q.log
