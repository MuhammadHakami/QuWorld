#!/bin/bash
# §45j q/b arm: QHR2V_s1 (QHR2 recipe + vacuum-field inputs) vs the Spark control QHR2c_s1, all VT frames.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45j_qb.log
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
echo "[s45j qb] start $(date -Is)" >> $LOG
[ -f $L/head_qb_QHR2V_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_VAC=1 STRAT_OUT_TAG=_QHR2V QW_GPU_GB=16 runs/capped.sh hQHR2V 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHR2V_s1.log 2>&1 || { echo "CRASH QHR2V $(date -Is)" >> $LOG; exit 1; }
echo "QHR2V_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2c_s1 head_qb_QHR2V_s1 >> $LOG 2>&1 || echo "CRASH vt qb $(date -Is)" >> $LOG
echo "S45J_QB_DONE $(date -Is)" >> $LOG
