#!/bin/bash
# doc 35 §41: RRNCO hybrid pilot (1 epoch x 10k, seed 1, full test files) on the Spark.
cd /home/mohhakami/Documents/paper/quworld/baseline/rrnco; export PYTHONUNBUFFERED=1; L=../../runs/routefinder/s41.log
flock ../../runs/strat/gpu.lock env QW_GPU_GB=30 ../../runs/capped.sh rrnco-h 30 ../../.venv-vrp/bin/python ../../paper/vrp/rrnco_hybrid.py \
  --seed 1 --epochs 1 --train-size 10000 --lr ${LR:-1e-3} >> $L 2>&1 || echo "CRASH $(date -Is)" >> $L
echo "S41_PILOT_DONE lr ${LR:-1e-3} $(date -Is)" >> $L
