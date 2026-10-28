#!/bin/bash
# doc 35 §29: (a) MHD divergence-free B on the gated model (VALID decides, TEST once); (b) fusion 3-D register on the SoTA trunk.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 30; done
QW_GPU_GB=16 timeout 7200 runs/capped.sh s29a 32 .venv/bin/python paper/alloc/mhd_divfree.py --variant vanilla >> runs/alloc/s29.log 2>&1 \
  || echo "CRASH s29a $(date -Is)" >> $S
echo "S29A_DONE $(date -Is)" >> $S
while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 30; done
QW_GPU_GB=16 timeout 28800 runs/capped.sh s29b 32 .venv/bin/python paper/nas/sophelio_3d.py >> runs/alloc/s29.log 2>&1 \
  || echo "CRASH s29b $(date -Is)" >> $S
echo "S29B_DONE $(date -Is)" >> $S
