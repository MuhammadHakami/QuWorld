#!/bin/bash
# doc 35 §40y: isolated env-step cost calibration rerun (the first one hit RESOURCE_EXHAUSTED at vmap-8 value-and-gradient
# under XLA fraction 0.3) once the Spark is otherwise idle (after runs/s40y3_spark.sh).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
while pgrep -f "s40y[35]_spark[.]sh" > /dev/null || pgrep -f "pop_probe[.]py|o1_race[.]py|evo_race[.]py|run_submission[.]py" > /dev/null; do sleep 60; done
QW_GPU_GB=62 runs/capped.sh l2dcal 62 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/calibrate.py --round 3 --run 1 > runs/l2d/calibrate_spark.log 2>&1
