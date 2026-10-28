#!/bin/bash
# §42x (registered before any result): mixup for the shape heads. OOD shots (unusual combinations of familiar coil patterns, §42w) carry ~30%
# of head SSE; mixup (Beta(0.2, 0.2) convex pairs of frames) trains on combinations. J1RLM_s1 = J1RL recipe + STRAT_MIXUP=0.2, TRAIN only,
# seed 1, vs J1RL_s1 (vt_eval shape, all VT shots; ood_heads.py breakdown). Rule: adopt (J1RLMA x8 for a later F-build) iff mean-7 delta
# >= +0.0003 with CI > 0. Starts after §42v's read-out (one head training at a time next to the F7 build / step-4 VT).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42x.log
until grep -q "^S42Y_DONE" $L/s42y.log 2>/dev/null; do sleep 60; done     # rerun (12:05): after F7 + post-passes, next to the step-4 VT
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_boundary_v8_J1RLM_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_MIXUP=0.2 \
  STRAT_OUT_TAG=_J1RLM QW_GPU_GB=16 runs/capped.sh hJ1RLM 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || { echo "CRASH J1RLM $(date -Is)" >> $LOG; exit 1; }
echo "J1RLM_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLM_s1 >> $LOG 2>&1 || echo "CRASH vt shape $(date -Is)" >> $LOG
(cd paper/strat/fusion && ../../../.venv/bin/python ood_heads.py head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLM_s1) >> $LOG 2>&1 || echo "CRASH ood $(date -Is)" >> $LOG
echo "S42X_DONE $(date -Is)" >> $LOG
