#!/bin/bash
# RealPDE: the FNO corrector arms were guard-killed at 08:53 after arm "none"; arms.py skips finished (arm, seed) files,
# so resume lin / mlp30 / mr30ryd once the U-Net arms and the diversity ensemble are done (one GPU job at a time).
cd /home/mohhakami/Documents/paper/quworld/paper/realpde; export QW_GPU_GB=16 PYTHONUNBUFFERED=1; L=../../runs/realpde/arms_runner.log
until grep -q REALPDE_DIVERSITY_DONE ../../runs/realpde/diversity.log 2>/dev/null; do sleep 300; done
../../runs/capped.sh rp-arms-fno 20 ../../.venv/bin/python arms.py --trunk fno >> $L 2>&1 && echo "REALPDE_FNO_ARMS_DONE $(date -Is)" >> $L || echo "CRASH arms fno resume $(date -Is)" >> $L
