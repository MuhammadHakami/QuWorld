#!/bin/bash
# RealPDE: after the corrector arms, add the released U-Net-real checkpoint (a different training paradigm) and re-fit
# the VALID-weighted ensemble over U-Net-finetune, U-Net-real and FNO-finetune.
cd /home/mohhakami/Documents/paper/quworld/paper/realpde; export QW_GPU_GB=14 PYTHONUNBUFFERED=1; L=../../runs/realpde/diversity.log
until grep -q REALPDE_ARMS_DONE ../../runs/realpde/arms_runner.log 2>/dev/null; do sleep 300; done
../../runs/capped.sh rp-real 24 ../../.venv/bin/python eval_trunk.py --models unet --stage real >> $L 2>&1
../../runs/capped.sh rp-ens2 16 ../../.venv/bin/python ensemble.py --models unet unet:real fno >> $L 2>&1; echo "REALPDE_DIVERSITY_DONE $(date -Is)" >> $L
