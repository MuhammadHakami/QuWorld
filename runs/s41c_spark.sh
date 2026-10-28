#!/bin/bash
# doc 35 §41c Fusion psi-model screen (stratified 3-fold CV on 900 TRAIN shots, paper/strat/fusion/cv_trunk.py; VALID/TEST
# untouched). Control = P90C0 (psi-only loss, 90 epochs, clip 1). Arms: +hist (long-window input history, hist_feats.py),
# E300 (300 epochs), and the scalar side: Q4E90 (+hist) for q95/betaN.
cd /home/mohhakami/Documents/paper/quworld/paper/strat/fusion; export PYTHONUNBUFFERED=1
PY=../../../.venv/bin/python
for arm in "C41_P90C0|{\"epochs\":90,\"clip\":1.0,\"w_scal\":0.0}" \
           "H41_P90C0h|{\"epochs\":90,\"clip\":1.0,\"w_scal\":0.0,\"hist\":true}" \
           "Q41_Q4E90h|{\"depth\":4,\"epochs\":90,\"hist\":true}" \
           "E41_P300C0|{\"epochs\":300,\"clip\":1.0,\"w_scal\":0.0}" \
           "E41_P300C0h|{\"epochs\":300,\"clip\":1.0,\"w_scal\":0.0,\"hist\":true}"; do
  name=${arm%%|*}; cfg=${arm#*|}
  QW_GPU_GB=60 ../../../runs/capped.sh $name 60 $PY cv_trunk.py $name "$cfg" > ../../../runs/strat/fusion/cv_$name.log 2>&1
done
