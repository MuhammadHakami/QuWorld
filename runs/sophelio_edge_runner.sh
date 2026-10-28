#!/bin/bash
# Sophelio shot-edge ablation (Sep 23): 4 arms x 2 seeds on the trial-5 config, VALID only (paper/sophelio/edge_ablation.py).
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=12 SOPHELIO_SCORE_WORKERS=4 PYTHONUNBUFFERED=1; S=runs/sophelio_sota/status
runs/capped.sh soph-edge 30 .venv/bin/python paper/sophelio/edge_ablation.py >> runs/sophelio_sota/edge_ablation.log 2>&1 \
  && echo "EDGE_ABLATION_DONE $(date -Is)" >> $S || echo "CRASH edge ablation $(date -Is)" >> $S
