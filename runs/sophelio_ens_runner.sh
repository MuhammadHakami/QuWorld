#!/bin/bash
# Sophelio SoTA push: after NAS stage 1, seed replicas of the top-3 trunks, then greedy ensemble selection on VALID.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=16 SOPHELIO_SCORE_WORKERS=4 PYTHONUNBUFFERED=1; S=runs/sophelio_sota/status
until grep -q NAS2_TRUNK_DONE "$S"; do sleep 300; done
runs/capped.sh soph-rep 30 .venv/bin/python paper/sophelio/replicas.py --study trunk1 --top 3 --seeds 1 2 >> runs/sophelio_sota/replicas.log 2>&1 || echo "CRASH replicas $(date -Is)" >> "$S"
runs/capped.sh soph-ens 30 .venv/bin/python paper/sophelio/ens_select.py --glob "$PWD/runs/nas/sophelio/trunks/trunk1_t*.pt" --top 10 >> runs/sophelio_sota/ens_select.log 2>&1 \
  && echo "ENS_SELECT_DONE: $(grep 'FULL VALID' runs/sophelio_sota/ens_select.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(date -Is)" >> "$S" || echo "CRASH ens_select $(date -Is)" >> "$S"
