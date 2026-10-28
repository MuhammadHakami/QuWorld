#!/bin/bash
# doc 35 §41: RRNCO hybrid for Table 1 - 3 seeds x 10 epochs x 10k, register-only, lr 1e-4 (pilot passed), Spark.
cd /home/mohhakami/Documents/paper/quworld/baseline/rrnco; export PYTHONUNBUFFERED=1; L=../../runs/routefinder/s41b.log
for s in 1 2 3; do
  [ -f ../../runs/routefinder/rrnco_hybrid_s${s}_e10_lr0.0001.json ] && continue
  flock ../../runs/strat/gpu.lock env QW_GPU_GB=30 ../../runs/capped.sh rrnco-h 30 ../../.venv-vrp/bin/python ../../paper/vrp/rrnco_hybrid.py \
    --seed $s --epochs 10 --train-size 10000 --lr 1e-4 >> $L 2>&1 || echo "CRASH s$s $(date -Is)" >> $L
done
echo "S41B_DONE $(date -Is)" >> $L
