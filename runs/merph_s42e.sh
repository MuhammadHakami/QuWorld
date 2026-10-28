#!/bin/bash
# Fusion §42e on merphistophline (registered in doc 35 before any result; moved from the Spark queue 16:52 to use the idle iGPU while the
# Learn2Design lane is settled): QHR2_s1 (QH recipe + qphys2) then J1R2_s1 (J1THS recipe + qphys2), TRAIN only. Yields to L2D: pauses
# while runs/strat/fusion/merph_yield.flag exists.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
while [ -f $L/merph_yield.flag ]; do sleep 60; done
[ -f $L/head_qb_QHR2_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_OUT_TAG=_QHR2 python3 paper/strat/fusion/head_boundary.py 1 qb > $L/QHR2_s1.log 2>&1
echo "QHR2_MERPH_DONE $(date -Is)" >> $L/QHR2_s1.log
while [ -f $L/merph_yield.flag ]; do sleep 60; done
[ -f $L/head_boundary_v8_J1R2_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_OUT_TAG=_J1R2 \
  python3 paper/strat/fusion/head_boundary.py 1 > $L/J1R2_s1.log 2>&1
echo "J1R2_MERPH_DONE $(date -Is)" >> $L/J1R2_s1.log
