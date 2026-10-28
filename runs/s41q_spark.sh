#!/bin/bash
# Fusion §41q learning curve (Spark, same machine as the 6 existing J1 seeds): J1 heads trained on 8/9 of the TRAIN shots
# (STRAT_FOLD k/9: 11% of TRAIN shots held out), seeds 1 / 2, vs the full-TRAIN J1_s1 / J1_s2. The 89% -> 100% step estimates
# the gain of refitting on TRAIN + VALID + TEST (7,041 / 6,298 = 1.12x) for the final build. Starts after the L2D lane race.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while pgrep -f "[l]ane_race.py" > /dev/null; do sleep 60; done
for k in 1 2; do
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 \
    STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_FOLD=$((k - 1))/9 STRAT_OUT_TAG=_J1L$((k - 1)) QW_GPU_GB=40 runs/capped.sh s41q 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $k >> $L/s41q_spark.log 2>&1 || echo "CRASH J1L seed $k $(date -Is)" >> $L/s41q_spark.log
done
echo "S41Q_SPARK_DONE $(date -Is)" >> $L/s41q_spark.log
