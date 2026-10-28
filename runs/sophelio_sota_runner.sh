#!/bin/bash
# Sophelio SoTA push (Sep 23): cache -> full-data classical baseline v1 -> joint classical+quantum NAS (Optuna 5, MO).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export SOPHELIO_SCORE_WORKERS=6
S=runs/sophelio_sota/status
until grep -q "diii_d_train done" runs/sophelio_sota/extract_train.log 2>/dev/null; do sleep 60; done
echo "extract done: $(grep 'done' runs/sophelio_sota/extract_train.log | tail -1) $(date -Is)" >> "$S"
.venv/bin/python -c "import sys; sys.path.insert(0,'paper/sophelio'); import sota; sota.build_cache()" > runs/sophelio_sota/cache.log 2>&1 \
  && echo "cache built: $(tail -1 runs/sophelio_sota/cache.log) $(date -Is)" >> "$S" || { echo "CRASH cache $(date -Is)" >> "$S"; exit 1; }
[ -f runs/sophelio_sota/v1.json ] || { .venv/bin/python paper/sophelio/sota_train.py --tag v1 --score-val 40 > runs/sophelio_sota/v1.log 2>&1 \
  && echo "v1: $(grep valid_official runs/sophelio_sota/v1.log | tail -1 | cut -c1-230) $(date -Is)" >> "$S" || echo "CRASH v1 $(date -Is)" >> "$S"; }
echo "nas start $(date -Is)" >> "$S"
.venv/bin/python paper/nas/sophelio_nas.py --trials 48 --study nas1 > runs/nas/sophelio_nas1.log 2>&1 \
  && echo "NAS_DONE: $(tail -1 runs/nas/sophelio_nas1.log | cut -c1-200) $(date -Is)" >> "$S" || echo "CRASH nas $(date -Is)" >> "$S"
