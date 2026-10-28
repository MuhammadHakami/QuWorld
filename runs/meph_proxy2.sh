#!/bin/bash
# Fusion §41q FAST PROXY v2 (v1 = 1/3 shots x 15 epochs had control-seed spread 0.0018, 12x the full-run sd - unusable): ALL TRAIN
# shots, small head (width 1024, depth 2), 30 epochs (~1/30 of the J1 FLOPs). Step 1: control seeds 1-3 (noise); step 2: thz
# seeds 1-3 (calibration vs the full-run +0.00096); then canon / hfull / aux / cs. Tag P2<arm>.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion; PY=${PY:-.venv-vrp/bin/python}; SEEDS=${SEEDS:-"1 2 3"}
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=30 STRAT_WIDTH=1024 STRAT_DEPTH=2 STRAT_INPUTS=magfix"
for arm in ctrl: thz:STRAT_THZ=1 canon:STRAT_CANON=1 hfull:STRAT_HFULL=1 aux:STRAT_AUX_PSI=1 cs:STRAT_CANON=2; do
  n=${arm%%:*}; x=${arm#*:}
  for s in $SEEDS; do
    [ -f $L/head_boundary_v8_P2${n}_s$s.npz ] && continue
    env $E $x STRAT_OUT_TAG=_P2$n $PY paper/strat/fusion/head_boundary.py $s > $L/P2${n}_s$s.log 2>&1
  done
done
echo "PROXY2_DONE $(date -Is)" >> $L/P2ctrl_s1.log
