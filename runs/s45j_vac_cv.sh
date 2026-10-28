#!/bin/bash
# Fusion §45j proxy CV: vacuum-field inputs (STRAT_VAC=1) vs the §45f base folds (same harness), one run at a time.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_EPOCHS=30"
for k in 0 1 2 3 4; do
  [ -f $L/head_boundary_v8_P6vacF${k}_s1_oof.npz ] && continue
  env $E STRAT_VAC=1 STRAT_CV=$k/5 STRAT_OUT_TAG=_P6vacF$k QW_GPU_GB=20 runs/capped.sh s45jcv 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/P6vacF${k}.log 2>&1 \
    && echo "P6vacF${k}_DONE $(date -Is)" >> $L/s45j_cv.log || echo "CRASH P6vac F$k $(date -Is)" >> $L/s45j_cv.log
done
CV_PFX=P6 .venv/bin/python paper/strat/fusion/cv_report.py base vac >> $L/s45j_cv.log 2>&1; echo "S45J_CV_DONE $(date -Is)" >> $L/s45j_cv.log
