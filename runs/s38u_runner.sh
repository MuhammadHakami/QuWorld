#!/bin/bash
# doc 35 §38u: 5-fold J1 heads (4096x4, magfix, seed 1) on TRAIN shots -> out-of-fold residuals -> register shot-bias stage.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
for k in 0 1 2 3 4; do
  [ -f $L/head_boundary_v8_J1F${k}_s1_oof.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 \
    STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_OUT_TAG=_J1F$k STRAT_FOLD=$k/5 QW_GPU_GB=40 runs/capped.sh s38u 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 \
    >> $L/s38u.log 2>&1 || echo "CRASH fold $k $(date -Is)" >> $L/s38u.log
done
(cd paper/strat/fusion && SB_OOF=1 flock ../../../runs/strat/gpu.lock ../../../.venv/bin/python shot_bias.py --arm reg --seed 1) >> $L/s38u.log 2>&1
echo "S38U_DONE $(date -Is)" >> $L/s38u.log
