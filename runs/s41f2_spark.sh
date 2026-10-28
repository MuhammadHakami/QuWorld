#!/bin/bash
# doc 35 §41f part 2: the scalar side (q95 / betaN trunk config Q4E90) with fixed-coordinate Thomson replacing the old block;
# control = the stored Q4E90 CV result (q95 0.9653, betaN 0.9755). Queued behind runs/s41c2_spark.sh.
cd /home/mohhakami/Documents/paper/quworld/paper/strat/fusion; export PYTHONUNBUFFERED=1
PY=../../../.venv/bin/python
while pgrep -f "[s]41c2_spark.sh" > /dev/null; do sleep 60; done
QW_GPU_GB=50 ../../../runs/capped.sh Q41r 50 $PY cv_trunk.py Q41r_Q4E90thzonly '{"depth":4,"epochs":90,"thz":true,"thomson":false}' > ../../../runs/strat/fusion/cv_Q41r_Q4E90thzonly.log 2>&1
