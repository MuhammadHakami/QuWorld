#!/bin/bash
# doc 35 §40y on the Spark after the O2 evolution races: (1) isolated env-step cost calibration; (2) rung 2 P1 per topology:
# the K 8 population (vmap B 8 value-and-gradient; XLA fraction 0.4) next to its K 1 control (0.15), one topology at a time
# (three concurrent B 8 evolution processes hit the memguard limit at 05:46); (3) rung 1b, three races at 0.2.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "o1_race[.]py|evo_race[.]py" > /dev/null || pgrep -f "s40y2_spark[.]sh" > /dev/null; do sleep 60; done
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 QW_GPU_GB=40 runs/capped.sh l2dcal 40 $PY paper/l2d/calibrate.py --round 3 --run 1 > runs/l2d/calibrate_spark.log 2>&1
for r in 1 8 6; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 QW_GPU_GB=50 runs/capped.sh l2dp1k8r$r 50 $PY paper/l2d/pop_probe.py --round 3 --run $r --arm naadam \
    --k 8 --s 30 --rounds 12 --tag p1nk8 > runs/l2d/pop_p1nk8_r3run$r.log 2>&1 &
  sleep 120
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 QW_GPU_GB=20 runs/capped.sh l2dp1k1r$r 20 $PY paper/l2d/pop_probe.py --round 3 --run $r --arm naadam \
    --k 1 --replace 0 --s 30 --rounds 96 --tag p1nk1 > runs/l2d/pop_p1nk1_r3run$r.log 2>&1 &
  wait
done
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
for r in 1 8 6; do
  QW_GPU_GB=25 runs/capped.sh l2do1br$r 25 $PY paper/l2d/o1_race.py --round 3 --run $r --units 380 \
    --arms n-cadam+b n-ademamix+b n-soap+b n-adopt+b > runs/l2d/o1b_spark_r3run$r.log 2>&1 &
  sleep 90
done
wait
