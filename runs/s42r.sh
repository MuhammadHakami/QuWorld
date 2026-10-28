#!/bin/bash
# Fusion §42r (registered before any result): QHR2L_s1 = QHR2 recipe (QH + qphys2) + liphys (Te^1.5 current-profile proxies, §42p),
# TRAIN only, seed 1, vs QHR2_s1 on all VT shots. Rule: adopt (QHAR2L all data) iff delta >= +0.0002 with CI > 0 (as §42i).
# Training lane: after §42j's training (<= 1 training + 1 VT job).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42r.log
# (16:29: starts at once - the VT lane was dropped, so the GPU slot is free)
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2L_s1.pt ] || env $E STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_QHR2L QW_GPU_GB=30 runs/capped.sh qQHR2L 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || { echo "CRASH QHR2L $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2_s1 head_qb_QHR2L_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S42R_DONE $(date -Is)" >> $LOG
