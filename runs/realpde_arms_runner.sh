#!/bin/bash
# RealPDE classical+quantum low-mode correctors (Sep 23): cache TRAIN predictions (FNO now; U-Net after its TEST eval),
# then paper/realpde/arms.py per trunk (arms none / lin / mlp30 / mr30ryd, 3 seeds).
cd /home/mohhakami/Documents/paper/quworld/paper/realpde; export QW_GPU_GB=16 PYTHONUNBUFFERED=1
L=../../runs/realpde/arms_runner.log; R=../../runs/capped.sh; PY=../../.venv/bin/python
$R rp-pred 20 $PY predict_split.py --model fno --split train >> $L 2>&1 || echo "CRASH predict fno train $(date -Is)" >> $L
$R rp-arms 20 $PY arms.py --trunk fno >> $L 2>&1 || echo "CRASH arms fno $(date -Is)" >> $L
while ps -eo args | grep -q "[e]val_trunk.py --models"; do sleep 120; done
$R rp-pred 20 $PY predict_split.py --model unet --split train >> $L 2>&1 || echo "CRASH predict unet train $(date -Is)" >> $L
$R rp-arms 20 $PY arms.py --trunk unet >> $L 2>&1 || echo "CRASH arms unet $(date -Is)" >> $L
echo "REALPDE_ARMS_DONE $(date -Is)" >> $L
