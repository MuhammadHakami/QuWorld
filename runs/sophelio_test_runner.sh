#!/bin/bash
# Sophelio single TEST scoring (rule in paper/sophelio/score_test.py), after the three VALID ensemble selections.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=12 SOPHELIO_SCORE_WORKERS=6 PYTHONUNBUFFERED=1; S=runs/sophelio_sota/status
until grep -qE "ENS_SELECT_MLP_DONE|CRASH ens_select mlp" $S; do sleep 300; done
runs/capped.sh soph-test 30 .venv/bin/python paper/sophelio/score_test.py >> runs/sophelio_sota/test_scores.log 2>&1 \
  && echo "SOPHELIO_TEST_DONE: $(grep '\[test\]' runs/sophelio_sota/test_scores.log | tr '\n' ' ' | cut -c1-500) $(date -Is)" >> $S || echo "CRASH sophelio test $(date -Is)" >> $S
