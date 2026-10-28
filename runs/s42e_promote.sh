#!/bin/bash
# Fusion §42e promotion (rule met, better than §42a): QHAR2 s1-3 = all-data q/b heads (QHA recipe) + qphys2 (scalings + coil / Ip ratios).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42e_promote.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 1 2 3; do
  [ -f $L/head_qb_QHAR2_s$s.pt ] || env $E STRAT_QPHYS=2 STRAT_OUT_TAG=_QHAR2 QW_GPU_GB=30 runs/capped.sh s42eq 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s qb >> $LOG 2>&1 || echo "CRASH QHAR2 s$s $(date -Is)" >> $LOG
done
echo "S42E_PROMOTE_DONE $(date -Is)" >> $LOG
