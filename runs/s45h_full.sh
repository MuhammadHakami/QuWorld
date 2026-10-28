#!/bin/bash
# §45h full-size: Huber (delta 0.3) shape head J1RLH_s1 vs J1RL_s1, then q/b QHR2H_s1 vs a Spark control QHR2c_s1 (all VT shots).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45h_full.log
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
echo "[s45h] start $(date -Is)" >> $LOG
[ -f $L/head_boundary_v8_J1RLH_s1.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_HUBER=0.3 \
  STRAT_OUT_TAG=_J1RLH QW_GPU_GB=20 runs/capped.sh hJ1RLH 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1RLH_s1.log 2>&1 || { echo "CRASH J1RLH $(date -Is)" >> $LOG; exit 1; }
echo "J1RLH_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLH_s1 >> $LOG 2>&1 || echo "CRASH vt shape $(date -Is)" >> $LOG
echo "S45H_SHAPE_DONE $(date -Is)" >> $LOG
for arm in QHR2c:0 QHR2H:0.3; do
  n=${arm%%:*}; h=${arm#*:}
  [ -f $L/head_qb_${n}_s1.pt ] || env $J STRAT_QPHYS=2 STRAT_HUBER=$h STRAT_OUT_TAG=_$n QW_GPU_GB=16 runs/capped.sh h$n 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/${n}_s1.log 2>&1 &
done; wait
echo "QB_TRAINED $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py qb head_qb_QHR2c_s1 head_qb_QHR2H_s1 head_qb_QHR2_s1 >> $LOG 2>&1 || echo "CRASH vt qb $(date -Is)" >> $LOG
echo "S45H_FULL_DONE $(date -Is)" >> $LOG
