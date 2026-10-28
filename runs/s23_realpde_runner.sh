#!/bin/bash
# doc 35 §23: RealPDEBench foil, frozen released U-Net, low-mode corrector arms (existing protocol), 3 seeds, predictions kept.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
until grep -q S23_TORAX_DONE runs/alloc/status; do sleep 30; done
while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 24 ]; do sleep 30; done
# predictions (6.3 GB per arm and split) are kept for seed 1 only: the disk has ~35 GB free
QW_GPU_GB=12 timeout 14400 runs/capped.sh s23rp 32 .venv/bin/python paper/realpde/arms.py --trunk unet --arms q30-geo=3d q30-geo=3d-mlp \
  --seeds 1 --save-pred >> runs/s23_realpde.log 2>&1 || echo "CRASH s23 realpde s1 $(date -Is)" >> runs/alloc/status
QW_GPU_GB=12 timeout 14400 runs/capped.sh s23rp 32 .venv/bin/python paper/realpde/arms.py --trunk unet --arms q30-geo=3d q30-geo=3d-mlp \
  --seeds 2 3 >> runs/s23_realpde.log 2>&1 || echo "CRASH s23 realpde s23 $(date -Is)" >> runs/alloc/status
echo "S23_REALPDE_DONE $(date -Is)" >> runs/alloc/status
