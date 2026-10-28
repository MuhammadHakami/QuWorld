#!/bin/bash
# doc 35 §38o: head capacity J1 (4096x4) / J2 (8192x3), corrected inputs, seed 1; then head_screen.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
for arm in "J1 4096 4" "J2 8192 3"; do set -- $arm
  [ -f $L/head_boundary_v8_$1_s1.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=$2 \
    STRAT_DEPTH=$3 STRAT_INPUTS=magfix STRAT_OUT_TAG=_$1 QW_GPU_GB=40 runs/capped.sh s38o 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 \
    >> $L/s38o.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38o.log
done
.venv/bin/python paper/strat/fusion/head_screen.py "head_boundary_v8_J1_s1" "head_boundary_v8_J2_s1" "head_boundary_v8_G1_s1" >> $L/s38o.log 2>&1
echo "S38O_DONE $(date -Is)" >> $L/s38o.log
