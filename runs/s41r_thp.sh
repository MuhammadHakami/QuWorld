#!/bin/bash
# Fusion §41r: thp block (Thomson on psi_N of the predicted psi). Public smoke with psiF, then all cache rows: raw channels + psiF
# features (TRAIN-only trunks -> VALID / TEST out-of-sample for the screen).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion/s41r_thp.log
QW_GPU_GB=30 runs/capped.sh s41rthp1 40 .venv/bin/python paper/strat/fusion/thp_feats.py feat --psi psiF --public >> $L 2>&1 || { echo "S41R_THP_FAIL public $(date -Is)" >> $L; exit 1; }
.venv/bin/python paper/strat/fusion/thp_feats.py raw --workers 8 >> $L 2>&1 || { echo "S41R_THP_FAIL raw $(date -Is)" >> $L; exit 1; }
QW_GPU_GB=30 runs/capped.sh s41rthp2 40 .venv/bin/python paper/strat/fusion/thp_feats.py feat --psi psiF >> $L 2>&1 || { echo "S41R_THP_FAIL all $(date -Is)" >> $L; exit 1; }
echo "S41R_THP_DONE $(date -Is)" >> $L
