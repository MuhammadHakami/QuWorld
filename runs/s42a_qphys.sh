#!/bin/bash
# Fusion §42a (registered before any result): QHR_s1 = QH recipe + qphys block vs QH_s1 on all VT shots. Starts after §41y-b's VT run.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42a.log
until grep -q "^S41Y_VT_DONE" $L/s41y_vt.log 2>/dev/null; do sleep 60; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR_s1.pt ] || env $J STRAT_QPHYS=1 STRAT_OUT_TAG=_QHR QW_GPU_GB=40 runs/capped.sh s42a 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || echo "CRASH QHR $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHR_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S42A_DONE $(date -Is)" >> $LOG
