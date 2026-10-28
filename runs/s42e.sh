#!/bin/bash
# Fusion §42e (registered before any result): coil-current / |Ip| ratio inputs (qphys2). QHR2_s1 (QH + qphys2) vs QH_s1 [and QHR_s1],
# then J1R2_s1 (J1THS recipe + qphys2) vs J1THS_s1; all TRAIN only, read out on all VT shots. Starts after §42a (QHR) finishes.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42e.log
until grep -q "^S42A_DONE" $L/s42a.log 2>/dev/null; do sleep 60; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHR2_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_OUT_TAG=_QHR2 QW_GPU_GB=30 runs/capped.sh s42eq 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || echo "CRASH QHR2 $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHR_s1 head_qb_QHR2_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
[ -f $L/head_boundary_v8_J1R2_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_OUT_TAG=_J1R2 QW_GPU_GB=30 \
  runs/capped.sh s42ej 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || echo "CRASH J1R2 $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1R2_s1 >> $LOG 2>&1 || echo "CRASH vt shape" >> $LOG
echo "S42E_DONE $(date -Is)" >> $LOG
