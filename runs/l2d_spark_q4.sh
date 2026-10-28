#!/bin/bash
# Spark (alone, GPU lock): v3b (submission defaults) on the remaining retired topologies 2, 7, 9, 10 (3,000 evaluations),
# after the ranker - completes the ten-topology dev table for the submitted version.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PY=baseline/Learn2Design-2026/.venv/bin/python; L=runs/l2d/spark_q.log
while pgrep -f "[d]onor_ranker.py" > /dev/null; do sleep 30; done
exec 9>runs/strat/gpu.lock; flock 9
for r in 2 7 9 10; do
  QW_GPU_GB=30 runs/capped.sh l2d 30 $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v3b > runs/l2d/sub_v3b_run$r.log 2>&1 || echo "CRASH v3b run $r $(date -Is)" >> $L
done
echo "L2D_SPARK_Q4_DONE $(date -Is)" >> $L
