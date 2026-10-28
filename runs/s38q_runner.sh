#!/bin/bash
# doc 35 §38q: q95/betaN head QH (4096x4, magfix) seed 1, then qb_eval arms on the VALID sample.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
[ -f $L/head_qb_QH_s1.npz ] || flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 \
  STRAT_INPUTS=magfix STRAT_OUT_TAG=_QH QW_GPU_GB=40 runs/capped.sh s38q 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb >> $L/s38q.log 2>&1
QF="s38g_QF.pt+s38g_QF-s1.pt+s38g_QF-s3.pt"
STRAT_TAG=_s38q flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/qb_eval.py "$QF@entry" "head:head_qb_QH_s1" "head:head_qb_QH_s1@entry" "$QF+head:head_qb_QH_s1@entry" >> $L/s38q.log 2>&1
echo "S38Q_DONE $(date -Is)" >> $L/s38q.log
