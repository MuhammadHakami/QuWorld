#!/bin/bash
# Fusion §41t screen (registered in doc 35 before any result): J1THF_s1 (thz + hfull + fut) vs J1THS_s1, QHF_s1 (QH + fut) vs QH_s1;
# Spark, seed 1, TRAIN only, full-size recipes; vt_eval.py on all VALID + TEST frames. After the §41r screen.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41t_screen.log
until grep -q FUT_DONE $L/fut_feats.log 2>/dev/null && grep -q S41R_SCREEN_DONE $L/s41r_screen.log 2>/dev/null; do sleep 30; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHF_s1.pt ] || flock runs/strat/gpu.lock env $J STRAT_FUT=1 STRAT_OUT_TAG=_QHF QW_GPU_GB=40 runs/capped.sh s41tQHF 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || echo "CRASH QHF $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHF_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
[ -f $L/head_boundary_v8_J1THF_s1.pt ] || flock runs/strat/gpu.lock env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_FUT=1 \
  STRAT_OUT_TAG=_J1THF QW_GPU_GB=40 runs/capped.sh s41tJ1THF 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || echo "CRASH J1THF $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1THF_s1 >> $LOG 2>&1 || echo "CRASH vt shape" >> $LOG
echo "S41T_SCREEN_DONE $(date -Is)" >> $LOG
