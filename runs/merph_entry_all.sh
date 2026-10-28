#!/bin/bash
# Fusion §41q F1 (merphistophline, after J1A seed 6): the 7 entry-ensemble members (+ the register member's stage-1 trunk) on all
# 7,041 shots with their stored configs (paper/strat/fusion/entry_alldata.py) -> runs/nas/sophelio/trunks_all/.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1A6_DONE $L/J1A_s6.log 2>/dev/null; do sleep 60; done
python3 paper/strat/fusion/entry_alldata.py > $L/entry_all.log 2>&1
echo "ENTRY_ALL_RUNNER_DONE $(date -Is)" >> $L/entry_all.log
