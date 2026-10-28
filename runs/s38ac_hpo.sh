#!/bin/bash
# doc 35 §38ac: Optuna classical + quantum NAS / HPO for the QPI-KAN shape head (Spark).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
flock runs/strat/gpu.lock env QW_GPU_GB=40 runs/capped.sh s38ac 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --trials 40 --frac 0.25 --epochs 6 \
  --study qpikan_v1 >> $L/s38ac_hpo.log 2>&1
echo "S38AC_HPO_DONE $(date -Is)" >> $L/s38ac_hpo.log
