#!/bin/bash
# Fusion §41r screen (registered in doc 35 before any result): J1THP_s1 (thz + hfull + thp_psiF) vs J1THS_s1 (thz + hfull), then
# QHP_s1 (QH + thp_psiF) vs QH_s1; Spark, seed 1, TRAIN only, full-size recipes; read-out vt_eval.py on all VALID + TEST frames.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41r_screen.log
until grep -q S41R_THP_DONE $L/s41r_thp.log 2>/dev/null; do grep -q S41R_THP_FAIL $L/s41r_thp.log 2>/dev/null && exit 1; sleep 30; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
for arm in J1THP:psiF J1THS:; do
  n=${arm%%:*}; p=${arm##*:}
  [ -f $L/head_boundary_v8_${n}_s1.pt ] || flock runs/strat/gpu.lock env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_THP=$p \
    STRAT_OUT_TAG=_$n QW_GPU_GB=40 runs/capped.sh s41r$n 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || echo "CRASH $n $(date -Is)" >> $LOG
done
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1THP_s1 >> $LOG 2>&1 || echo "CRASH vt shape" >> $LOG
[ -f $L/head_qb_QHP_s1.pt ] || flock runs/strat/gpu.lock env $J STRAT_THP=psiF STRAT_OUT_TAG=_QHP QW_GPU_GB=40 runs/capped.sh s41rQHP 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || echo "CRASH QHP $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHP_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S41R_SCREEN_DONE $(date -Is)" >> $LOG
