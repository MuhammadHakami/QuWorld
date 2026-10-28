#!/bin/bash
# §45h q/b arm (relaunch): QHR2Hu_s1 = QHR2 recipe + Huber 0.3 (the tag QHR2H was already taken by the Oct 11 hfull head, so s45h_full.sh
# skipped training); the Spark control QHR2c_s1 keeps training from s45h_full.sh. Then vt_eval qb (all VT shots).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45h_full.log
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2Hu_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_HUBER=0.3 STRAT_OUT_TAG=_QHR2Hu QW_GPU_GB=16 runs/capped.sh hQHR2Hu 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHR2Hu_s1.log 2>&1 || { echo "CRASH QHR2Hu $(date -Is)" >> $LOG; exit 1; }
until [ -f $L/head_qb_QHR2c_s1.pt ]; do sleep 30; done; sleep 30
echo "QB_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2c_s1 head_qb_QHR2Hu_s1 head_qb_QHR2_s1 >> $LOG 2>&1 || echo "CRASH vt qb $(date -Is)" >> $LOG
echo "S45H_FULL_DONE $(date -Is)" >> $LOG
