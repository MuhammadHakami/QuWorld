#!/bin/bash
# doc 35 §41p: after the v9 R3-8 run, D3 lane race on R3-8 (6,000 steps, 7 lanes) on merphistophline.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
while pgrep -f "[r]un_submission.py --round 3 --run 8" > /dev/null; do sleep 60; done
baseline/Learn2Design-2026/.venv-rocm/bin/python paper/l2d/lane_race.py --round 3 --run 8 --steps 6000 > runs/l2d/lanes_d3_r3run8.log 2>&1
