#!/bin/bash
# RealPDE (Sep 23 16:10): headline stack first, FNO corrector arms after. (1) cache the U-Net + linear low-mode corrector
# (seed 1, pre-declared) predictions on VALID/TEST; (2) VALID-weighted ensemble of corrected U-Net, U-Net-real, FNO;
# (3) resume the FNO arms (guard-killed at 08:53 after 'none'; arms.py skips finished arm/seed files).
cd /home/mohhakami/Documents/paper/quworld/paper/realpde
export QW_GPU_GB=16 PYTHONUNBUFFERED=1; L=../../runs/realpde/diversity.log; A=../../runs/realpde/arms_runner.log; R=../../runs/capped.sh; PY=../../.venv/bin/python
$R rp-savecorr 32 $PY arms.py --trunk unet --arms lin --seeds 1 --save-pred >> $A 2>&1 || echo "CRASH save corrected $(date -Is)" >> $L
QW_GPU_GB=14 $R rp-ens3 16 $PY ensemble.py --models unet-lin-s1 unet:real fno >> $L 2>&1; echo "REALPDE_STACK_DONE $(date -Is)" >> $L
$R rp-arms-fno 20 $PY arms.py --trunk fno >> $A 2>&1 && echo "REALPDE_FNO_ARMS_DONE $(date -Is)" >> $A || echo "CRASH arms fno resume $(date -Is)" >> $A
