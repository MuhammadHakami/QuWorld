#!/bin/bash
# Spark (alone, GPU lock): §40n v3b screen (adaptive pool: 64 donors, widen to 512 when the best start >= 1.5) on runs 5
# and 6; runs 1/8 under v3b are v2b (no widening), runs 3/4 are v3 (widening), both measured elsewhere.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python; L=runs/l2d/spark_q.log
exec 9>runs/strat/gpu.lock; flock 9
for r in 5 6; do
  QW_GPU_GB=30 runs/capped.sh l2d 30 $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v3b \
    --kw '{"k": 64, "k_wide": 512, "widen_if": 1.5, "lr_steps": [0.01, 0.003], "step_patience": 500}' > runs/l2d/sub_v3b_run$r.log 2>&1 || echo "CRASH v3b run $r $(date -Is)" >> $L
done
echo "L2D_SPARK_Q3_DONE $(date -Is)" >> $L
