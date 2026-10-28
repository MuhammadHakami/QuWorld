#!/bin/bash
# Sophelio ensemble selections (Sep 23 16:00), sequential: (1) classical pool (NAS trunks + seed/edge/ew2 variants),
# (2) classical pool + register-coupled models, (3) classical pool + their matched-MLP twins. VALID only.
# (First attempt crashed: loaders sized the network from the feature mask, missing the 2 edge inputs; now len(mu).)
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=16 SOPHELIO_SCORE_WORKERS=4 PYTHONUNBUFFERED=1
S=runs/sophelio_sota/status; T=$PWD/runs/nas/sophelio/trunks; R=runs/capped.sh; PY=.venv/bin/python
$R soph-ens 30 $PY paper/sophelio/ens_select.py --glob "$T/trunk1_t*.pt" --top 10 >> runs/sophelio_sota/ens_select.log 2>&1 \
  && echo "ENS_SELECT_DONE: $(grep 'FULL VALID' runs/sophelio_sota/ens_select.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(date -Is)" >> $S || echo "CRASH ens_select $(date -Is)" >> $S
for arm in reg mlp; do
  $R soph-ens-$arm 30 $PY paper/sophelio/ens_select.py --glob "$T/trunk1_t*.pt" "$T/coupled_c*-${arm}_seed*.pt" --top 12 --tag _$arm >> runs/sophelio_sota/ens_select_$arm.log 2>&1 \
    && echo "ENS_SELECT_${arm^^}_DONE: $(grep 'FULL VALID' runs/sophelio_sota/ens_select_$arm.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(date -Is)" >> $S || echo "CRASH ens_select $arm $(date -Is)" >> $S
done
