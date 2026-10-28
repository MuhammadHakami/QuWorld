#!/bin/bash
# doc 35 §41c part 2 (queued behind runs/s41c_spark.sh): psi-model levers at the 300-epoch scale, psi-only, clip 1.
# G0 = pixel MSE (gamma 0: the pooled-R2 metric itself), W4K = width 4096 (failed only at 30 epochs), D0 = dropout 0.
cd /home/mohhakami/Documents/paper/quworld/paper/strat/fusion; export PYTHONUNBUFFERED=1
PY=../../../.venv/bin/python
while pgrep -f "[s]41c_spark.sh" > /dev/null; do sleep 60; done
for arm in "G41_P300C0g0|{\"epochs\":300,\"clip\":1.0,\"w_scal\":0.0,\"gamma\":0.0}" \
           "W41_P300C0w4k|{\"epochs\":300,\"clip\":1.0,\"w_scal\":0.0,\"width\":4096}" \
           "D41_P300C0d0|{\"epochs\":300,\"clip\":1.0,\"w_scal\":0.0,\"drop\":0.0}"; do
  name=${arm%%|*}; cfg=${arm#*|}
  QW_GPU_GB=60 ../../../runs/capped.sh $name 60 $PY cv_trunk.py $name "$cfg" > ../../../runs/strat/fusion/cv_$name.log 2>&1
done
