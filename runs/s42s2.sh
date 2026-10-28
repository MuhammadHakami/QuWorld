#!/bin/bash
# §42s follow-up (registered before any seed-2 result): QHR2S_s2 vs QHR2_s1 on all VT shots (s1 gave +0.000185 [-0.000049, +0.000451]).
# Rule: QHAR2S enters the next F-build iff the mean of the s1 and s2 paired deltas >= +0.0002 and both > 0. No auto-promotion.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42s2.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2S_s2.pt ] || env $E STRAT_QPHYS=2 STRAT_QSHAPE=1 STRAT_OUT_TAG=_QHR2S QW_GPU_GB=30 runs/capped.sh qQHR2S2 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 2 qb >> $LOG 2>&1 || { echo "CRASH QHR2S s2 $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2_s1 head_qb_QHR2S_s2 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S42S2_DONE $(date -Is)" >> $LOG
