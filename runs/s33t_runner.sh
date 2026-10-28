#!/bin/bash
# doc 35 §33t: one registered TEST pass of the 20 frozen §33 checkpoints (scorer refuses any rerun; lock tag s33t-lock).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; L=runs/s33/s33t.log
until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 24 ]; do sleep 60; done
QW_GPU_GB=8 timeout 3600 runs/capped.sh s33t 16 .venv/bin/python runs/s33t_score.py >> $L 2>&1 \
  && echo "S33T_DONE $(date -Is)" >> $S || echo "CRASH s33t (see runs/s33/s33t.log; claim/journal decide recovery) $(date -Is)" >> $S
