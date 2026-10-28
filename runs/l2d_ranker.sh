#!/bin/bash
# Spark: §40j hybrid donor ranker (classical encoder + MultiRegister 3x10 Rydberg), LOTO over the 10 retired topologies,
# fresh-topology scans as extra training data. GPU lock held.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/l2d/ranker.log
flock runs/strat/gpu.lock env QW_GPU_GB=12 runs/capped.sh l2d-ranker 16 .venv/bin/python paper/l2d/donor_ranker.py --k 64 --epochs 150 --seed 1 >> $L 2>&1 || echo "CRASH $(date -Is)" >> $L
