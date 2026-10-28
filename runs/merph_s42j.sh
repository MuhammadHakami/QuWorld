#!/bin/bash
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
while [ -f $L/merph_yield.flag ]; do sleep 60; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=120 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2E_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_OUT_TAG=_QHR2E python3 paper/strat/fusion/head_boundary.py 1 qb > $L/QHR2E_s1.log 2>&1
echo "QHR2E_MERPH_DONE $(date -Is)" >> $L/QHR2E_s1.log
