#!/bin/bash
# Fusion §41q PROXY CV (user: cross-validation instead of seeds): small head (width 1024, depth 2, 30 epochs) on stratified shot folds
# (STRAT_CV k/5: Ip sign x shot-mean li quartile), seed 1, out-of-fold metric on all TRAIN shots; folds split across machines
# (FOLDS env), every arm on the same folds -> paired per fold. Arms: ctrl, thz (calibration), thz + hfull, thz + input noise, thz 60 ep,
# thz replace, thz + end weight. Tag P3<arm>F<k>.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion; PY=${PY:-.venv-vrp/bin/python}; FOLDS=${FOLDS:-"0 1 2"}
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_INPUTS=magfix"
for arm in ctrl:"STRAT_EPOCHS=30" thz:"STRAT_EPOCHS=30 STRAT_THZ=1" thf:"STRAT_EPOCHS=30 STRAT_THZ=1 STRAT_HFULL=1" \
           tnoise:"STRAT_EPOCHS=30 STRAT_THZ=1 STRAT_XNOISE=0.05" t60:"STRAT_EPOCHS=60 STRAT_THZ=1" \
           trep:"STRAT_EPOCHS=30 STRAT_THZ=1 STRAT_NO_THOMSON=1" tend:"STRAT_EPOCHS=30 STRAT_THZ=1 STRAT_END_W=2"; do
  n=${arm%%:*}; x=${arm#*:}
  for k in $FOLDS; do
    [ -f $L/head_boundary_v8_P3${n}F${k}_s1_oof.npz ] && continue
    env $E $x STRAT_CV=$k/5 STRAT_OUT_TAG=_P3${n}F$k $PY paper/strat/fusion/head_boundary.py 1 > $L/P3${n}F${k}.log 2>&1
  done
done
echo "CV_DONE $(date -Is)" >> $L/P3ctrlF${FOLDS%% *}.log
