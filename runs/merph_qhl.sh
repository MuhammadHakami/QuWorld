#!/bin/bash
# Fusion §41q (merphistophline, after J1MR): q/b heads with log targets - QHL (log q95 / log betaN, plain MSE) and QHLW (log
# targets weighted by y) - QH recipe otherwise; vs the Spark QH_s1 (merph matched the Spark on J1: 0.98662 vs seeds 0.98634-0.98661).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1MR_DONE $L/J1MR_s1.log 2>/dev/null; do sleep 60; done
for arm in QHL:0 QHLW:1; do
  n=${arm%%:*}; w=${arm##*:}
  env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix \
    STRAT_QB_LOG=1 STRAT_QB_LOGW=$w STRAT_OUT_TAG=_$n python3 paper/strat/fusion/head_boundary.py 1 qb > $L/${n}_s1.log 2>&1
done
echo "QHL_DONE $(date -Is)" >> $L/QHL_s1.log
