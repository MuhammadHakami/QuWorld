#!/bin/bash
# Doc 35 §45c (Spark): v2 shape trees (all TRAIN frames, depth 10 / lr 0.05 / 6000 it) -> VT OOF blend study.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 GBDT_V2=1; L=runs/strat/fusion/s45c_gbdt_v2.log
QW_GPU_GB=30 runs/capped.sh gbdt2_dump 50 .venv/bin/python paper/strat/fusion/gbdt_shape.py dump >> $L 2>&1 || { echo "CRASH dump $(date -Is)" >> $L; exit 1; }
QW_GPU_GB=40 runs/capped.sh gbdt2_fit 60 .venv-tdc/bin/python paper/strat/fusion/gbdt_shape.py fit >> $L 2>&1 || { echo "CRASH fit $(date -Is)" >> $L; exit 1; }
QW_GPU_GB=30 runs/capped.sh gbdt2_eval 50 .venv/bin/python paper/strat/fusion/gbdt_shape.py eval >> $L 2>&1 || { echo "CRASH eval $(date -Is)" >> $L; exit 1; }
echo "S45C_DONE $(date -Is)" >> $L
