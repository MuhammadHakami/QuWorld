#!/bin/bash
# Fusion §42a promotion (rule met): QHAR s1-3 = all-data q/b heads (QHA recipe) + qphys physics scalings.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42a_promote.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 1 2 3; do
  [ -f $L/head_qb_QHAR_s$s.pt ] || env $E STRAT_QPHYS=1 STRAT_OUT_TAG=_QHAR QW_GPU_GB=30 runs/capped.sh s42aq 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s qb >> $LOG 2>&1 || echo "CRASH QHAR s$s $(date -Is)" >> $LOG
done
echo "S42A_PROMOTE_DONE $(date -Is)" >> $LOG
