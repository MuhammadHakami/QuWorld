#!/bin/bash
# Fusion §41t proxy CV (user: cross-validation): stratified 5-fold over all TRAIN shots, proxy head (1024 x 2, 30 epochs), seed 1;
# P5thf = thz + hfull (the deployed input set) vs P5fut = thz + hfull + fut; both on the Spark (paired within machine).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_INPUTS=magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_EPOCHS=30"
for k in 0 1 2 3 4; do
  for arm in thf: fut:"STRAT_FUT=1"; do
    n=${arm%%:*}; x=${arm#*:}
    [ -f $L/head_boundary_v8_P5${n}F${k}_s1_oof.npz ] && continue
    env $E $x STRAT_CV=$k/5 STRAT_OUT_TAG=_P5${n}F$k QW_GPU_GB=16 runs/capped.sh s41tcv 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/P5${n}F${k}.log 2>&1 \
      || echo "CRASH P5$n F$k" >> $L/s41t_cv.log
  done
done
CV_PFX=P5 .venv/bin/python paper/strat/fusion/cv_report.py thf fut >> $L/s41t_cv.log 2>&1
echo "S41T_CV_DONE $(date -Is)" >> $L/s41t_cv.log
