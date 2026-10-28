#!/bin/bash
# Fusion §41q F1 helper (merphistophline matches the Spark on J1 heads): J1A seed 6 (all data + fixed-coordinate Thomson, X_magfix),
# exactly the s41q_final.sh recipe; the .pt / .npz are copied to the Spark, whose runner then skips seed 6.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 \
  QW_MAGFIX=X_magfix STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_OUT_TAG=_J1A python3 paper/strat/fusion/head_boundary.py 6 > $L/J1A_s6.log 2>&1
echo "J1A6_DONE $(date -Is)" >> $L/J1A_s6.log
