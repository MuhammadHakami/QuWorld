#!/bin/bash
# Fusion §42g on merphistophline (registered before any result): QHH_s1 = QH recipe + full-rate history, after §42e's J1R2.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1R2_MERPH_DONE $L/J1R2_s1.log 2>/dev/null; do sleep 120; done
while [ -f $L/merph_yield.flag ]; do sleep 60; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHH_s1.pt ] || env $J STRAT_HFULL=1 STRAT_OUT_TAG=_QHH python3 paper/strat/fusion/head_boundary.py 1 qb > $L/QHH_s1.log 2>&1
echo "QHH_MERPH_DONE $(date -Is)" >> $L/QHH_s1.log
