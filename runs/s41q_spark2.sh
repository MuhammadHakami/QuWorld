#!/bin/bash
# Fusion §41q q/b heads with log targets (Spark, same machine as QH_s1): QHL = log q95 / log betaN (plain MSE in log space),
# QHLW = log targets weighted by y (between log- and q-space MSE). QH recipe otherwise (s38s: 4096x4, 60 epochs, clip 1, magfix).
# Read-out: q95 / betaN R2 on the VALID sample rows vs QH_s1. After the learning-curve heads.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q S41Q_SPARK_DONE $L/s41q_spark.log 2>/dev/null; do sleep 60; done
for arm in QHL:0 QHLW:1; do
  n=${arm%%:*}; w=${arm##*:}
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 \
    STRAT_INPUTS=magfix STRAT_QB_LOG=1 STRAT_QB_LOGW=$w STRAT_OUT_TAG=_$n QW_GPU_GB=40 runs/capped.sh s41q2 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $L/s41q_spark2.log 2>&1 || echo "CRASH $n $(date -Is)" >> $L/s41q_spark2.log
done
echo "S41Q_SPARK2_DONE $(date -Is)" >> $L/s41q_spark2.log
