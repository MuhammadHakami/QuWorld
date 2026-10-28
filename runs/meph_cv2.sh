#!/bin/bash
# Fusion §41q proxy CV batch 2 (after meph_cv.sh): thf60 (thz + hfull, 60 epochs) and thf at 90 epochs (thf90) vs thf - does longer
# training add to the history gain; trhf (thz replacing the old Thomson block + hfull). Same stratified folds, tag P3<arm>F<k>.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion; PY=${PY:-.venv-vrp/bin/python}; FOLDS=${FOLDS:-"0 1 2"}
until grep -q CV_DONE $L/P3ctrlF${FOLDS%% *}.log 2>/dev/null; do sleep 30; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_INPUTS=magfix STRAT_THZ=1 STRAT_HFULL=1"
for arm in thf60:"STRAT_EPOCHS=60" thf90:"STRAT_EPOCHS=90" trhf:"STRAT_EPOCHS=30 STRAT_NO_THOMSON=1"; do
  n=${arm%%:*}; x=${arm#*:}
  for k in $FOLDS; do
    [ -f $L/head_boundary_v8_P3${n}F${k}_s1_oof.npz ] && continue
    env $E $x STRAT_CV=$k/5 STRAT_OUT_TAG=_P3${n}F$k $PY paper/strat/fusion/head_boundary.py 1 > $L/P3${n}F${k}.log 2>&1
  done
done
echo "CV2_DONE $(date -Is)" >> $L/P3thf60F${FOLDS%% *}.log
