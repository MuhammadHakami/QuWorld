#!/bin/bash
# Fusion §41q F2 heads, speculative (proxy CV thz + hfull +0.00087, t 3.2; full-size J1TH pending on unreachable mephiles): J1AH = all data
# + thz + full-rate history (24 inputs), J1 recipe, seeds 1, 2 on merphistophline.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
for s in 7 8; do
  [ -f $L/head_boundary_v8_J1AH_s$s.pt ] && continue
  env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 \
    QW_MAGFIX=X_magfix STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_OUT_TAG=_J1AH python3 paper/strat/fusion/head_boundary.py $s > $L/J1AH_s$s.log 2>&1
done
echo "J1AH78_MERPH_DONE $(date -Is)" >> $L/J1AH_s7.log
