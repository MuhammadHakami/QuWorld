#!/bin/bash
# Sophelio joint classical+quantum NAS (restart after the Sep 23 02:26 OOM/reboot). Resumable (Optuna RDB).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export SOPHELIO_SCORE_WORKERS=6; export QW_GPU_GB=24
S=runs/sophelio_sota/status; echo "nas start $(date -Is)" >> "$S"
runs/capped.sh sophelio-nas 24 .venv/bin/python paper/nas/sophelio_nas.py --trials 48 --study nas1 >> runs/nas/sophelio_nas1.log 2>&1 \
  && echo "NAS_DONE: $(tail -1 runs/nas/sophelio_nas1.log | cut -c1-200) $(date -Is)" >> "$S" || echo "CRASH nas $(date -Is)" >> "$S"
