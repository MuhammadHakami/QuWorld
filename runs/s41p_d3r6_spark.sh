#!/bin/bash
# doc 35 §41p D3 lane race on R3-6 (better families exist: PoolHop 0.157), 4,500 steps, Spark alone.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
QW_GPU_GB=60 runs/capped.sh d3r6 60 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/lane_race.py --round 3 --run 6 --steps 4500 > runs/l2d/lanes_d3_r3run6.log 2>&1
