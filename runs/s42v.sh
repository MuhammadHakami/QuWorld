#!/bin/bash
# §42v (registered before any result): shape heads with the 4 X-point coordinates as auxiliary targets (weight 2; inference drops them).
# J1RLX_s1 = J1RL recipe + STRAT_AUX_XP=2 (TRAIN only, seed 1) vs J1RL_s1 (vt_eval shape, all VT shots). Rule: adopt (all-data J1RLXA x8
# for a later F-build) iff mean-7 delta >= +0.0003 with CI > 0. Starts after F5c's build (<= 2 GPU jobs with the step-4 VT).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42v.log
until grep -q "^F5C_DONE\|^F5C_FAIL\|^CRASH" $L/f5c.log 2>/dev/null; do sleep 60; done
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_boundary_v8_J1RLX_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_AUX_XP=2 \
  STRAT_OUT_TAG=_J1RLX QW_GPU_GB=30 runs/capped.sh hJ1RLX 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || { echo "CRASH J1RLX $(date -Is)" >> $LOG; exit 1; }
echo "J1RLX_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLX_s1 >> $LOG 2>&1 || echo "CRASH vt shape $(date -Is)" >> $LOG
echo "S42V_DONE $(date -Is)" >> $LOG
