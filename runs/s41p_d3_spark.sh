#!/bin/bash
# doc 35 §41p: after the v9 R3-1 run (one B=7 L2D job at a time on the Spark), D3 lane race on R3-1 (6,000 steps).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
while pgrep -f "[r]un_submission.py --round 3 --run 1 " > /dev/null; do sleep 60; done
QW_GPU_GB=55 runs/capped.sh d3r1 55 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/lane_race.py --round 3 --run 1 --steps 6000 > runs/l2d/lanes_d3_r3run1.log 2>&1
