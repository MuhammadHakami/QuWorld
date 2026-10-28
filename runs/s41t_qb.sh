#!/bin/bash
# Fusion §41t (q/b only; the shape arm J1THF was cancelled after the proxy CV: -0.00024, t -1.2): QHF_s1 (QH + fut) vs QH_s1, Spark,
# TRAIN only; vt_eval on all VALID + TEST frames incl. the last-frame betaN share. After the §41r screen.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41t_screen.log
until grep -q S41R_SCREEN_DONE $L/s41r_screen.log 2>/dev/null; do sleep 30; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_qb_QHF_s1.pt ] || flock runs/strat/gpu.lock env $J STRAT_FUT=1 STRAT_OUT_TAG=_QHF QW_GPU_GB=40 runs/capped.sh s41tQHF 40 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $LOG 2>&1 || echo "CRASH QHF $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QH_s1 head_qb_QHF_s1 >> $LOG 2>&1 || echo "CRASH vt qb" >> $LOG
echo "S41T_SCREEN_DONE $(date -Is)" >> $LOG
