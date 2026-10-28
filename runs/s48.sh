#!/bin/bash
# §48c (doc 35, registered before any result): 8-fold stratified search of the VisionHOPE + register hybrid.
# 1) control M8 on folds 0-5; 2) Optuna search on folds 0-2 (3 workers, 16 trials, fold-wise median pruning); 3) V / best / consensus
# on the held-out folds 3-5; report (amended 22:50, user: accelerate). The 60-epoch seeds of the winner follow the registered rule (separate step).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s48.log
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=24"; P=paper/strat/fusion/hpo_vq.py
echo "[s48] start $(date -Is)" >> $LOG
for fs in 0,3 1,4 2,5; do env $E runs/capped.sh s48c 36 .venv/bin/python $P ctrl $fs >> $L/s48_ctrl.log 2>&1 & sleep 20; done; wait
n=$(ls $L/quota/M8_f*.npz 2>/dev/null | wc -l); echo "[s48] control folds $n/8 $(date -Is)" >> $LOG; [ "$n" -ge 6 ] || { echo "S48_FAIL ctrl $(date -Is)" >> $LOG; exit 1; }
for w in 0 1 2; do env $E HPO_WORKER=$w runs/capped.sh s48w$w 36 .venv/bin/python $P search 20 > $L/s48_w$w.log 2>&1 & sleep 90; done; wait
echo "[s48] search done $(date -Is)" >> $LOG; .venv/bin/python $P report >> $LOG 2>&1
for c in V best consensus; do env $E runs/capped.sh s48k$c 36 .venv/bin/python $P confirm $c 3,4,5 > $L/s48_conf_$c.log 2>&1 & sleep 20; done; wait
.venv/bin/python $P report >> $LOG 2>&1
echo "S48_DONE $(date -Is)" >> $LOG
