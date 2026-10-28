#!/bin/bash
# F16K rebuild budget (rule 6): measured MLP-head throughput on the GB10, alone vs 4 concurrent, with the J1RLHA all-data recipe at reduced epochs
# (throwaway tags _TIMx; checkpoints deleted afterwards). Wall times -> runs/strat/fusion/budget_timing.log.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion/budget_timing.log; F=paper/strat/fusion
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix STRAT_ALL=1 STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_HUBER=0.3"
run() { env $E STRAT_EPOCHS=$1 STRAT_OUT_TAG=_TIM$2 .venv/bin/python $F/head_boundary.py 1 > runs/strat/fusion/tim_$2.log 2>&1; }
echo "[timing] start $(date -Is)" >> $L
t=$(date +%s); run 6 a1; echo "alone 6 epochs: $(( $(date +%s) - t )) s" >> $L
t=$(date +%s); run 12 a2; echo "alone 12 epochs: $(( $(date +%s) - t )) s" >> $L
t=$(date +%s); pids=(); for k in c1 c2 c3 c4; do run 12 $k & pids+=($!); done; for p in "${pids[@]}"; do wait "$p"; done; echo "4 concurrent x 12 epochs: $(( $(date +%s) - t )) s" >> $L
rm -f runs/strat/fusion/head_boundary_v8_TIM*_s1.pt
echo "TIMING_DONE $(date -Is)" >> $L
