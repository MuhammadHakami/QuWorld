#!/bin/bash
# Fusion §41q F2 heads on the Spark: J1AH (all data + thz + full-rate history; proxy CV +0.00087 vs thz, t 3.2) seeds 3-6 after the F1b
# public build (seeds 1-2 on merphistophline).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41q_f2heads.log
while pgrep -f "[g]pu_build.py --split public --tag F1b" > /dev/null || pgrep -f "[e]ntry_alldata" > /dev/null; do sleep 60; done
for s in 3 4 5 6; do
  [ -f $L/head_boundary_v8_J1AH_s$s.pt ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix \
    STRAT_ALL=1 QW_MAGFIX=X_magfix STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_OUT_TAG=_J1AH QW_GPU_GB=40 runs/capped.sh s41qf2 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s >> $LOG 2>&1 || echo "CRASH J1AH s$s $(date -Is)" >> $LOG
done
echo "S41Q_F2HEADS_DONE $(date -Is)" >> $LOG
