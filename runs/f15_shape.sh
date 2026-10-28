#!/bin/bash
# F15 shape heads (§45l): J1RLHA s1-8 = J1RLA recipe + Huber 0.3, all data, sequential.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f15.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
echo "F15_SHAPE_START $(date -Is)" >> $LOG
for s in 1 2 3 4 5 6 7 8; do
  [ -f $L/head_boundary_v8_J1RLHA_s$s.pt ] || env $E STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_HUBER=0.3 STRAT_OUT_TAG=_J1RLHA QW_GPU_GB=30 \
    runs/capped.sh hJ1RLHA 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s >> $L/J1RLHA.log 2>&1 || { echo "CRASH J1RLHA s$s $(date -Is)" >> $LOG; exit 1; }
  echo "J1RLHA_s${s}_TRAINED $(date -Is)" >> $LOG
done
echo "F15_SHAPE_DONE $(date -Is)" >> $LOG
