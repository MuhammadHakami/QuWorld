#!/bin/bash
# Fusion §41q proxy v2 batch 4 (after batch 3): on top of thz - trep (thz REPLACES the index-resampled Thomson block), tend (end-of-shot
# frames x2 weight; the worst li / volume frames are ramp-down), tw2048 (width 2048: capacity trend at this data size). Control = P2thz.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion; PY=${PY:-.venv-vrp/bin/python}; SEEDS=${SEEDS:-"1 2 3"}
until grep -q PROXY4_DONE $L/P2tdrop_s${SEEDS%% *}.log 2>/dev/null; do sleep 30; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=30 STRAT_DEPTH=2 STRAT_INPUTS=magfix STRAT_THZ=1"
for arm in trep:"STRAT_WIDTH=1024 STRAT_NO_THOMSON=1" tend:"STRAT_WIDTH=1024 STRAT_END_W=2" tw2048:"STRAT_WIDTH=2048"; do
  n=${arm%%:*}; x=${arm#*:}
  for s in $SEEDS; do
    [ -f $L/head_boundary_v8_P2${n}_s$s.npz ] && continue
    env $E $x STRAT_OUT_TAG=_P2$n $PY paper/strat/fusion/head_boundary.py $s > $L/P2${n}_s$s.log 2>&1
  done
done
echo "PROXY5_DONE $(date -Is)" >> $L/P2trep_s${SEEDS%% *}.log
