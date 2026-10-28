#!/bin/bash
# §45o: TRAIN-only canonical twins (STRAT_CANON=2: Ip-sign canonical currents + sign input) of the J1RL shape head and the QHR2 q/b head,
# concurrent beside F15; router read-out on all VT shots vs J1RL_s1 / QHR2c_s1 (Spark controls). Rule in doc 35 §45o.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45o.log
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
echo "[s45o] start $(date -Is)" >> $LOG
[ -f $L/head_boundary_v8_J1RLCS_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_CANON=2 \
  STRAT_OUT_TAG=_J1RLCS QW_GPU_GB=20 runs/capped.sh hJ1RLCS 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1RLCS_s1.log 2>&1 \
  && echo "J1RLCS_TRAINED $(date -Is)" >> $LOG || echo "CRASH J1RLCS $(date -Is)" >> $LOG &
[ -f $L/head_qb_QHR2CS_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_CANON=2 STRAT_OUT_TAG=_QHR2CS QW_GPU_GB=16 runs/capped.sh hQHR2CS 30 \
  .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHR2CS_s1.log 2>&1 && echo "QHR2CS_TRAINED $(date -Is)" >> $LOG || echo "CRASH QHR2CS $(date -Is)" >> $LOG &
wait
[ -f $L/head_qb_QHR2CS_s1.pt ] && .venv/bin/python paper/strat/fusion/router_eval.py qb head_qb_QHR2c_s1 head_qb_QHR2CS_s1 >> $LOG 2>&1
[ -f $L/head_boundary_v8_J1RLCS_s1.pt ] && .venv/bin/python paper/strat/fusion/router_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLCS_s1 >> $LOG 2>&1
echo "S45O_DONE $(date -Is)" >> $LOG
