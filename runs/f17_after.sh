#!/bin/bash
# F17 (doc 35 §45w): build once the §45s trainings are done (XPACS s3), compute-neutral form REGALL=1 XPALL=1. No upload here.
cd /home/mohhakami/Documents/paper/quworld; L=runs/strat/fusion
until grep -q "^S45S_TRAINED" $L/s45s.log; do sleep 120; done
for s in 1 2 3; do [ -f $L/head_xp_XPACS_s$s.pt ] || { echo "F17_FAIL missing XPACS_s$s $(date -Is)" >> $L/f17.log; exit 1; }; done
REGALL=1 XPALL=1 TAG=F17 runs/f17.sh
