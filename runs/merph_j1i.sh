#!/bin/bash
# Fusion §41q ipfix screen: J1 shape head (4096x4, v8 labels, 60 epochs; s38p recipe) seed 1 on X_ipfix (ROCm, merphistophline).
# Gate: head_screen J1I_s1 vs J1_s1 (magfix) on the VALID sample before retraining seeds 2-6.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_MAGFIX=X_ipfix; L=runs/strat/fusion
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 \
  STRAT_INPUTS=magfix STRAT_OUT_TAG=_J1I python3 paper/strat/fusion/head_boundary.py 1 > $L/J1I_s1.log 2>&1
echo "J1I_DONE $(date -Is)" >> $L/J1I_s1.log
