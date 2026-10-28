#!/bin/bash
# §42p follow-up (registered before any seed-2 result): J1RL_s2 (single seed) vs J1THS_s1, paired on all VT shots; s1 gave +0.000374
# [-0.000075, +0.000850]. Rule: production shape heads gain qphys2 + liphys iff the mean of the two single-seed paired deltas >= +0.0004
# and both deltas > 0 (an ensemble-vs-single comparison would be biased by the ensembling gain).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42p2.log
H="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_boundary_v8_J1RL_s2.pt ] || env $H STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_J1RL QW_GPU_GB=30 \
  runs/capped.sh qJ1RL2 40 .venv/bin/python paper/strat/fusion/head_boundary.py 2 >> $LOG 2>&1 || { echo "CRASH J1RL s2 $(date -Is)" >> $LOG; exit 1; }
echo "J1RL2_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1RL_s2 >> $LOG 2>&1 || echo "CRASH vt" >> $LOG
echo "S42P2_DONE $(date -Is)" >> $LOG
