#!/bin/bash
# Fusion §41q: same-machine control for J1I (merphistophline): J1 head seed 1 on the ORIGINAL magfix inputs, after J1I.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1I_DONE $L/J1I_s1.log 2>/dev/null; do sleep 60; done
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 \
  STRAT_INPUTS=magfix STRAT_OUT_TAG=_J1MR python3 paper/strat/fusion/head_boundary.py 1 > $L/J1MR_s1.log 2>&1
echo "J1MR_DONE $(date -Is)" >> $L/J1MR_s1.log
