#!/bin/bash
# doc 35 §39d on the Spark: route30-tw seeds 5 and 4 (mephiles runs 1-3), same protocol (bf16, 10 x 10k, lr 3e-4).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; O=runs/routefinder/bf16_spark_tw; mkdir -p $O
for seed in 5 4; do
  [ -f $O/eal_md_route30-tw_s$seed.json ] && continue
  (cd baseline/routefinder && flock ../../runs/strat/gpu.lock env QW_GPU_GB=40 ../../runs/capped.sh rf-tw 40 ../../.venv-vrp/bin/python \
    ../../paper/vrp/run_rf_eal.py route30-tw --setting md --seed $seed --epochs 10 --train-size 10000 --out ../../$O) >> $O/runner.log 2>&1 \
    || echo "CRASH s$seed $(date -Is)" >> $O/runner.log
done
echo "RF_TW_SPARK_DONE $(date -Is)" >> $O/runner.log
