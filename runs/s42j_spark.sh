#!/bin/bash
# §42j moved to the Spark (registered 2026-10-11T23:03; merph run stopped at ~60% per the user's lane rule): QHR2E_s1 = QHR2 recipe with
# 120 epochs (TRAIN only) vs QHR2_s1 on all VT shots. Rule: adopt (QHAR2E all-data) iff delta >= +0.0002 with CI > 0. Training lane: waits
# for the J1RL seed-2 training to finish (<= 1 training + 1 VT job).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42j_spark.log
until grep -q "^J1RL2_TRAINED\|^CRASH" $L/s42p2.log 2>/dev/null; do sleep 60; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2E_s1.pt ] || env $E STRAT_EPOCHS=120 STRAT_QPHYS=2 STRAT_OUT_TAG=_QHR2E QW_GPU_GB=30 runs/capped.sh qQHR2E 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || { echo "CRASH QHR2E $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2_s1 head_qb_QHR2E_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S42J_DONE $(date -Is)" >> $LOG
