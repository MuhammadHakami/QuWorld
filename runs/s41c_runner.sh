#!/bin/bash
# doc 35 §41c: safe RRNCO hybrid (gate + write-back penalty), seed 1 then (if within +0.05 pp) seeds 2-3, Spark.
cd /home/mohhakami/Documents/paper/quworld/baseline/rrnco; export PYTHONUNBUFFERED=1; L=../../runs/routefinder/s41c.log
run() { flock ../../runs/strat/gpu.lock env QW_GPU_GB=30 ../../runs/capped.sh rrnco-h 30 ../../.venv-vrp/bin/python ../../paper/vrp/rrnco_hybrid.py \
  --seed $1 --epochs 10 --train-size 10000 --lr 1e-4 --safe --lam 1 >> $L 2>&1 || echo "CRASH s$1 $(date -Is)" >> $L; }
run 1
ok=$(python3 -c "import json; d=json.load(open('../../runs/routefinder/rrnco_hybrid_s1_e10_lr0.0001_safe1.json')); print(int(d['hybrid_gap']['in'] <= d['classical_gap']['in'] + 0.05))" 2>/dev/null)
[ "$ok" = 1 ] && { run 2; run 3; }
echo "S41C_DONE gate=$ok $(date -Is)" >> $L
