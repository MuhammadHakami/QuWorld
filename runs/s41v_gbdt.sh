#!/bin/bash
# Fusion §41v: CatBoost q95 / betaN (CPU, 16 threads; GBDT allowed on CPU) -> VT predictions -> blend study (GPU inference).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion/s41v_gbdt.log
runs/capped.sh s41vgbdt 24 .venv-tdc/bin/python paper/strat/fusion/gbdt_qb.py fit >> $L 2>&1 || { echo "S41V_FAIL fit" >> $L; exit 1; }
CUDA_VISIBLE_DEVICES=0 QW_GPU_GB=10 .venv/bin/python paper/strat/fusion/gbdt_qb.py eval >> $L 2>&1 || { echo "S41V_FAIL eval" >> $L; exit 1; }
echo "S41V_DONE $(date -Is)" >> $L
