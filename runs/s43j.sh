#!/bin/bash
# §43j (registered before any result; user Oct 14: NAS / Optuna on sampled data + seed CV): register-head search with the J1RL physics
# inputs, space v2 around the deployed J1QPIpi recipe (enqueued as trial 0 = the baseline under the same protocol). Objective: mean R2
# of the 7 scalars on 10% of TRAIN shots held out by shot (VT never used); trials fit 25% of the remaining TRAIN rows for 6 epochs;
# QW_QFAST=1 for the search only (ranking proxy). Gate for final retrains: best trial - trial 0 >= +0.002 held-out R2, then the
# top config x 3 seeds on the per-qubit path (seed CV) vs J1QPIpi s1-3 on VT.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43j.log
echo "[s43j] start $(date -Is)" >> $LOG
env STRAT_CUDA=0 QW_QFAST=1 QW_MAGFIX=X_magfix QW_GPU_GB=30 STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 runs/capped.sh hpo43j 50 \
  .venv/bin/python paper/strat/fusion/hpo_qpikan.py --space v2 --study qpikan_rl_v2 --trials 40 --frac 0.25 --epochs 6 >> $LOG 2>&1 \
  && echo "S43J_SEARCH_DONE $(date -Is)" >> $LOG || echo "CRASH s43j search $(date -Is)" >> $LOG
