#!/bin/bash
# Sophelio stage-2 confirmation + matched ensembles (Sep 23 13:40). After the coupling search and the classical ensemble
# selection: top-3 coupling configs x {register, matched MLP} x 3 seeds on the frozen trunk (paper/nas/sophelio_confirm.py),
# then greedy ensemble selection twice over the classical pool plus (a) the register-coupled or (b) the MLP-coupled models.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=16 SOPHELIO_SCORE_WORKERS=4 PYTHONUNBUFFERED=1
S=runs/sophelio_sota/status; T=$PWD/runs/nas/sophelio/trunks; R=runs/capped.sh; PY=.venv/bin/python
until grep -qE "NAS2_COUPLING_DONE|CRASH nas2 coupling" $S; do sleep 300; done
$R soph-confirm 30 $PY paper/nas/sophelio_confirm.py >> runs/nas/sophelio_confirm.log 2>&1 \
  && echo "CONFIRM_DONE: $(grep 'vs matched MLP' runs/nas/sophelio_confirm.log | tr '\n' ' ' | cut -c1-400) $(date -Is)" >> $S || echo "CRASH confirm $(date -Is)" >> $S
until grep -qE "^ENS_SELECT_DONE|CRASH ens_select" $S; do sleep 300; done       # one ensemble selection at a time
for arm in reg mlp; do
  $R soph-ens-$arm 30 $PY paper/sophelio/ens_select.py --glob "$T/trunk1_t*.pt" "$T/coupled_c*-${arm}_seed*.pt" --top 12 --tag _$arm >> runs/sophelio_sota/ens_select_$arm.log 2>&1 \
    && echo "ENS_SELECT_${arm^^}_DONE: $(grep 'FULL VALID' runs/sophelio_sota/ens_select_$arm.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(date -Is)" >> $S || echo "CRASH ens_select $arm $(date -Is)" >> $S
done
