#!/bin/bash
# Figure 2 RouteFinder row: after the doc 35 §23 RouteFinder chain (vanilla, 3-D register, matched 3-D MLP), draw the
# saved policies' routes on the fixed first test instance (paper/vrp/dump_rows_s23.py). No training.
cd /home/mohhakami/Documents/paper/quworld; S=runs/alloc/status
until grep -q S23_RF_DONE $S; do sleep 120; done
while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done
(cd baseline/routefinder && QW_GPU_GB=12 timeout 3600 ../../runs/capped.sh rfrows 30 ../../.venv-vrp/bin/python ../../paper/vrp/dump_rows_s23.py) \
  >> runs/s23_rf.log 2>&1 && echo "S23_RFROWS_DONE $(date -Is)" >> $S || echo "CRASH s23 rf rows $(date -Is)" >> $S
