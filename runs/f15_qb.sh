#!/bin/bash
# F15 q/b heads (§45l): QHAR2Hu s1-3 = QHAR2 recipe + Huber 0.3, all data, sequential (runs beside f15_shape.sh).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f15.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 1 2 3; do
  [ -f $L/head_qb_QHAR2Hu_s$s.pt ] || env $E STRAT_QPHYS=2 STRAT_HUBER=0.3 STRAT_OUT_TAG=_QHAR2Hu QW_GPU_GB=20 runs/capped.sh qQHAR2Hu 30 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s qb >> $L/QHAR2Hu.log 2>&1 || { echo "CRASH QHAR2Hu s$s $(date -Is)" >> $LOG; exit 1; }
  echo "QHAR2Hu_s${s}_TRAINED $(date -Is)" >> $LOG
done
echo "F15_QB_DONE $(date -Is)" >> $LOG
