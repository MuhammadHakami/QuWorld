#!/bin/bash
# doc 35 §38y: shot-phase heads (P1 shot-phase inputs, P2 + end weight 4, QP q/b + shot-phase), seed 1, then screens. Spark (after the ranker).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while pgrep -f "[d]onor_ranker.py" > /dev/null; do sleep 30; done
B="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_GPU_GB=40"
run() { flock runs/strat/gpu.lock env $B $2 STRAT_OUT_TAG=_$1 runs/capped.sh s38y 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 $3 >> $L/s38y.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38y.log; }
[ -f $L/head_boundary_v8_P1_s1.npz ] || run P1 "STRAT_EDGE=1"
[ -f $L/head_boundary_v8_P2_s1.npz ] || run P2 "STRAT_EDGE=1 STRAT_END_W=4"
[ -f $L/head_qb_QP_s1.npz ] || run QP "STRAT_EDGE=1" qb
.venv/bin/python paper/strat/fusion/head_screen.py "head_boundary_v8_P1_s1" "head_boundary_v8_P2_s1" "head_boundary_v8_J1_s1" >> $L/s38y.log 2>&1
STRAT_TAG=_s38y flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/qb_eval.py "head:head_qb_QP_s1" "head:head_qb_QH_s1" >> $L/s38y.log 2>&1
echo "S38Y_DONE $(date -Is)" >> $L/s38y.log
