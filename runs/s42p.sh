#!/bin/bash
# Fusion §42p (registered before any result): J1RL_s1 = J1THS recipe (thz + hfull) + qphys2 + liphys (Te^1.5 current-profile proxies),
# TRAIN only, seed 1; vt_eval shape vs J1THS_s1 and J1R2_s1 (thz + hfull + qphys2) on all VT shots. <= 3 GPU jobs: starts after PQ2A s1-3.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42p.log
until grep -q "^S42M_DONE" $L/s42m.log 2>/dev/null; do sleep 60; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_boundary_v8_J1RL_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_J1RL QW_GPU_GB=30 \
  runs/capped.sh s42p 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || { echo "CRASH J1RL $(date -Is)" >> $LOG; exit 1; }
echo "J1RL_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1R2_s1 head_boundary_v8_J1RL_s1 >> $LOG 2>&1 || echo "CRASH vt shape" >> $LOG
echo "S42P_DONE $(date -Is)" >> $LOG
