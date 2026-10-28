#!/bin/bash
# Spark (alone, GPU lock): §40n v3 screen, runs 1 and 8 (after l2d_spark_q.sh).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python; L=runs/l2d/spark_q.log
while pgrep -f "[l]2d_spark_q.sh" > /dev/null; do sleep 30; done
exec 9>runs/strat/gpu.lock; flock 9
for r in 1 8; do
  QW_GPU_GB=30 runs/capped.sh l2d 30 $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v3 \
    --kw '{"k": 512, "lr_steps": [0.01, 0.003], "step_patience": 500}' > runs/l2d/sub_v3_run$r.log 2>&1 || echo "CRASH v3 run $r $(date -Is)" >> $L
done
echo "L2D_SPARK_Q2_DONE $(date -Is)" >> $L
