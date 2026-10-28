#!/bin/bash
# Fusion §42s (registered before any result): QHR2S_s1 = QHR2 recipe (QH + qphys2) + qshape (out-of-fold plasma-shape features, §42s),
# TRAIN only, seed 1, vs QHR2_s1 on all VT shots. Rule: adopt iff delta >= +0.0002 with CI > 0.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42s.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2S_s1.pt ] || env $E STRAT_QPHYS=2 STRAT_QSHAPE=1 STRAT_OUT_TAG=_QHR2S QW_GPU_GB=30 runs/capped.sh qQHR2S 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || { echo "CRASH QHR2S $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2_s1 head_qb_QHR2S_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S42S_DONE $(date -Is)" >> $LOG
