#!/bin/bash
# Sophelio: EMA arms after the 4-arm edge ablation (VALID only).
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=12 SOPHELIO_SCORE_WORKERS=4 PYTHONUNBUFFERED=1; S=runs/sophelio_sota/status
until grep -qE "EDGE_ABLATION_DONE|CRASH edge ablation" $S; do sleep 120; done
runs/capped.sh soph-ema 30 .venv/bin/python paper/sophelio/edge_ablation.py --arms ema edge-ew2-ema >> runs/sophelio_sota/edge_ablation.log 2>&1 \
  && echo "EDGE_EMA_DONE $(date -Is)" >> $S || echo "CRASH edge ema $(date -Is)" >> $S
