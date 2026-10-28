#!/bin/bash
# Sophelio two-stage NAS (Sep 23 restructure): stage 1 classical trunk (20 trials, successive halving), stage 2 coupling
# (latent / srr register configs) on the frozen best trunk (24 trials, MO S vs atoms x layers). Resumable (Optuna RDB).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 SOPHELIO_SCORE_WORKERS=6 QW_GPU_GB=24
S=runs/sophelio_sota/status; L=runs/nas/sophelio_nas2.log
echo "nas2 trunk stage start $(date -Is)" >> "$S"
runs/capped.sh nas-trunk 30 .venv/bin/python paper/nas/sophelio_nas.py --stage trunk --trials 20 --study trunk1 >> "$L" 2>&1 \
  && echo "NAS2_TRUNK_DONE: $(grep 'trunk best' "$L" | tail -1) $(date -Is)" >> "$S" || { echo "CRASH nas2 trunk $(date -Is)" >> "$S"; exit 1; }
runs/capped.sh nas-coupling 30 .venv/bin/python paper/nas/sophelio_nas.py --stage coupling --trials 24 --study coupling1 --trunk-study trunk1 >> "$L" 2>&1 \
  && echo "NAS2_COUPLING_DONE: $(grep 'Pareto' "$L" | tail -1 | cut -c1-200) $(date -Is)" >> "$S" || echo "CRASH nas2 coupling $(date -Is)" >> "$S"
