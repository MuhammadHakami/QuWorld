#!/bin/bash
# RealPDE (Sep 23 12:30): after the FNO arms resume, cache the U-Net + linear low-mode corrector (seed 1, the pre-declared
# seed) predictions on VALID/TEST, then refit the VALID-weighted ensemble over corrected U-Net, U-Net-real and FNO.
cd /home/mohhakami/Documents/paper/quworld/paper/realpde; export QW_GPU_GB=16 PYTHONUNBUFFERED=1; L=../../runs/realpde/diversity.log
until grep -qE "REALPDE_FNO_ARMS_DONE|CRASH arms fno resume" ../../runs/realpde/arms_runner.log 2>/dev/null; do sleep 300; done
../../runs/capped.sh rp-savecorr 32 ../../.venv/bin/python arms.py --trunk unet --arms lin --seeds 1 --save-pred >> ../../runs/realpde/arms_runner.log 2>&1 || echo "CRASH save corrected $(date -Is)" >> $L
QW_GPU_GB=14 ../../runs/capped.sh rp-ens3 16 ../../.venv/bin/python ensemble.py --models unet-lin-s1 unet:real fno >> $L 2>&1; echo "REALPDE_STACK_DONE $(date -Is)" >> $L
