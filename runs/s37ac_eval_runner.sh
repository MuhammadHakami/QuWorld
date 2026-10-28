#!/bin/bash
# doc 35 §37ac: only if §37ab passed TEST. Q-bkl3 with the 6-seed v8 mean vs the 3-seed v8 mean on the sample.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37AC_HEADS_DONE\|^CRASH s37ac' runs/alloc/status && grep -q '^S37AB_TEST_DONE\|^CRASH s37ab test' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37ac\|^CRASH s37ab test' runs/alloc/status && exit 1
python3 -c "import json,sys; d=json.load(open('runs/strat/fusion/bw_test_kw_v8.json')); sys.exit(0 if d['delta_S'] > 0 else 1)" || { echo "S37AC_SKIPPED (§37ab TEST not positive) $(date -Is)" >> runs/alloc/status; exit 0; }
cd paper/strat/fusion
STRAT_HEAD_SEEDS=1,2,3,4,5,6 STRAT_LABEL_SUFFIX=_v8 STRAT_WORKERS=8 timeout 7200 ../../../.venv/bin/python bw_valid.py Q-bkl3 > ../../../runs/strat/fusion/bw_valid_Q-bkl3_v8_6s.log 2>&1
grep -q '^S37S_VALID' ../../../runs/strat/fusion/bw_valid_Q-bkl3_v8_6s.log && echo "S37AC_VALID_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37ac valid $(date -Is)" >> ../../../runs/alloc/status
