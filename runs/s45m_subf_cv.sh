#!/bin/bash
# Fusion §45m proxy CV: sub-frame window statistics (STRAT_SUBF=1) vs the §45f base folds (same harness), one run at a time.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q "^SUBF_DONE" $L/subf_feats.log 2>/dev/null; do
  grep -q -E "Traceback|non-finite" $L/subf_feats.log 2>/dev/null && { echo "CRASH subf features $(date -Is)" >> $L/s45m_cv.log; exit 1; }
  sleep 30
done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_EPOCHS=30"
for k in 0 1 2 3 4; do
  [ -f $L/head_boundary_v8_P6subfF${k}_s1_oof.npz ] && continue
  env $E STRAT_SUBF=1 STRAT_CV=$k/5 STRAT_OUT_TAG=_P6subfF$k QW_GPU_GB=20 runs/capped.sh s45mcv 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/P6subfF${k}.log 2>&1 \
    && echo "P6subfF${k}_DONE $(date -Is)" >> $L/s45m_cv.log || echo "CRASH P6subf F$k $(date -Is)" >> $L/s45m_cv.log
done
CV_PFX=P6 .venv/bin/python paper/strat/fusion/cv_report.py base subf >> $L/s45m_cv.log 2>&1; echo "S45M_CV_DONE $(date -Is)" >> $L/s45m_cv.log
