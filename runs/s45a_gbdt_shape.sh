#!/bin/bash
# Doc 35 §45a (Spark): CatBoost shape heads (7 consistency scalars) on the J1RL input set; dump -> GPU fit -> VT blend study (5-fold shot OOF).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion/s45a_gbdt_shape.log
QW_GPU_GB=30 runs/capped.sh gbdt_dump 40 .venv/bin/python paper/strat/fusion/gbdt_shape.py dump >> $L 2>&1 || { echo "CRASH dump $(date -Is)" >> $L; exit 1; }
QW_GPU_GB=30 runs/capped.sh gbdt_fit 40 .venv-tdc/bin/python paper/strat/fusion/gbdt_shape.py fit >> $L 2>&1 || { echo "CRASH fit $(date -Is)" >> $L; exit 1; }
QW_GPU_GB=30 runs/capped.sh gbdt_eval 40 .venv/bin/python paper/strat/fusion/gbdt_shape.py eval >> $L 2>&1 || { echo "CRASH eval $(date -Is)" >> $L; exit 1; }
echo "S45A_DONE $(date -Is)" >> $L
