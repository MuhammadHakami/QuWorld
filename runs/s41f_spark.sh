#!/bin/bash
# doc 35 §41f: Thomson on fixed physical coordinates (thz_feats.py) in the psi-model CV screen; control C41 (P90C0).
# T41 = P90C0 + thz (added to the old Thomson block); T41r = P90C0 with thz replacing the old index-resampled Thomson block.
cd /home/mohhakami/Documents/paper/quworld/paper/strat/fusion; export PYTHONUNBUFFERED=1
PY=../../../.venv/bin/python
for arm in "T41_P90C0thz|{\"epochs\":90,\"clip\":1.0,\"w_scal\":0.0,\"thz\":true}" \
           "T41r_P90C0thzonly|{\"epochs\":90,\"clip\":1.0,\"w_scal\":0.0,\"thz\":true,\"thomson\":false}"; do
  name=${arm%%|*}; cfg=${arm#*|}
  QW_GPU_GB=50 ../../../runs/capped.sh $name 50 $PY cv_trunk.py $name "$cfg" > ../../../runs/strat/fusion/cv_$name.log 2>&1
done
