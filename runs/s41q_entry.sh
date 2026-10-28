#!/bin/bash
# Fusion §41q F1: the all-data entry ensemble (paper/strat/fusion/entry_alldata.py) on the Spark after the classical F1 components
# (merphistophline's ROCm runtime aborts on it twice: rocr AqlPacket vector assertion at start-up).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q S41Q_FINAL_DONE $L/s41q_final.log 2>/dev/null; do sleep 60; done
flock runs/strat/gpu.lock env QW_GPU_GB=40 runs/capped.sh s41qe 40 .venv/bin/python paper/strat/fusion/entry_alldata.py >> $L/entry_all_spark.log 2>&1
echo "S41Q_ENTRY_DONE $(date -Is)" >> $L/entry_all_spark.log
