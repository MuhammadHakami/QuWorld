#!/bin/bash
# doc 35 §29c: MHD shift average + divergence-free B (VALID decides; TEST once), after §29a.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
until grep -q S29A_DONE $S; do sleep 60; done
while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 30; done
QW_GPU_GB=16 timeout 14400 runs/capped.sh s29c 32 .venv/bin/python paper/alloc/mhd_divfree.py --variant vanilla --tta >> runs/alloc/s29.log 2>&1 \
  || echo "CRASH s29c $(date -Is)" >> $S
echo "S29C_DONE $(date -Is)" >> $S
