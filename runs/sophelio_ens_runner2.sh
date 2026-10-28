#!/bin/bash
# Sophelio SoTA push v2 (Sep 23 12:05): replaces sophelio_ens_runner.sh + sophelio_ema_runner.sh. One Sophelio GPU job at a
# time (the coupling-stage NAS runs beside it). The shot-edge ablation showed psi-loss up-weighting of the first / last 10 %
# of each shot (ew2) beats the base config on both seeds (VALID 0.9858 / 0.9856 vs 0.9762 / 0.9837), so the ensemble pool
# gets ew2 seeds of the top stage-1 configs instead of unweighted replicas; then greedy selection over every saved trunk.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=12 SOPHELIO_SCORE_WORKERS=4 PYTHONUNBUFFERED=1
S=runs/sophelio_sota/status; L=runs/sophelio_sota/edge_ablation.log; R=runs/capped.sh; PY=.venv/bin/python
until grep -qE "EDGE_ABLATION_DONE|CRASH edge ablation" $S; do sleep 120; done
$R soph-ema 30 $PY paper/sophelio/edge_ablation.py --arms ema edge-ew2-ema >> $L 2>&1 || echo "CRASH edge ema $(date -Is)" >> $S
$R soph-ew5 30 $PY paper/sophelio/edge_ablation.py --trial 5 --arms ew2 --seeds 3 4 5 6 >> $L 2>&1 || echo "CRASH ew2 t5 $(date -Is)" >> $S
$R soph-ew19 30 $PY paper/sophelio/edge_ablation.py --trial 19 --arms ew2 --seeds 1 2 >> $L 2>&1 || echo "CRASH ew2 t19 $(date -Is)" >> $S
$R soph-ew7 30 $PY paper/sophelio/edge_ablation.py --trial 7 --arms ew2 --seeds 1 >> $L 2>&1 || echo "CRASH ew2 t7 $(date -Is)" >> $S
echo "EW2_POOL_DONE $(date -Is)" >> $S
$R soph-ens 30 $PY paper/sophelio/ens_select.py --glob "$PWD/runs/nas/sophelio/trunks/trunk1_t*.pt" --top 10 >> runs/sophelio_sota/ens_select.log 2>&1 \
  && echo "ENS_SELECT_DONE: $(grep 'FULL VALID' runs/sophelio_sota/ens_select.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(date -Is)" >> $S || echo "CRASH ens_select $(date -Is)" >> $S
