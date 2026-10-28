#!/bin/bash
# Spark (alone, GPU lock): §40r v4 (log-length) runs 1 and 3 on CUDA - mephiles (registered for these) is unreachable.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python; L=runs/l2d/spark_q.log
exec 9>runs/strat/gpu.lock; flock 9
for r in 1 3; do
  QW_GPU_GB=30 runs/capped.sh l2d 30 $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v4spark --kw '{"loglen": true}' > runs/l2d/sub_v4spark_run$r.log 2>&1 || echo "CRASH v4 run $r $(date -Is)" >> $L
done
echo "L2D_SPARK_Q5_DONE $(date -Is)" >> $L
