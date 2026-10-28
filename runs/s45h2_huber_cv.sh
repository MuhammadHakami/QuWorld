#!/bin/bash
# Fusion §45h proxy CV: Huber loss (delta 1.0 and 0.3) vs the §45f base folds (same harness), one run at a time.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_EPOCHS=30"
for k in 0 1 2 3 4; do
  for arm in hub01:0.1 hub003:0.03; do
    n=${arm%%:*}; dl=${arm#*:}
    [ -f $L/head_boundary_v8_P6${n}F${k}_s1_oof.npz ] && continue
    env $E STRAT_HUBER=$dl STRAT_CV=$k/5 STRAT_OUT_TAG=_P6${n}F$k QW_GPU_GB=20 runs/capped.sh s45h2cv 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/P6${n}F${k}.log 2>&1 \
      && echo "P6${n}F${k}_DONE $(date -Is)" >> $L/s45h2_cv.log || echo "CRASH P6$n F$k $(date -Is)" >> $L/s45h2_cv.log
  done
done
CV_PFX=P6 .venv/bin/python paper/strat/fusion/cv_report.py base hub03 hub01 hub003 >> $L/s45h2_cv.log 2>&1; echo "S45H2_CV_DONE $(date -Is)" >> $L/s45h2_cv.log
