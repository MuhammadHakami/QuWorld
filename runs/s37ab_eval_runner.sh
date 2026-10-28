#!/bin/bash
# doc 35 §37ab: Q-bkl3 with the v8 heads vs Q-bkl3 with the v5 heads on the sample, after the v8 heads; then head accuracy.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37AB_HEADS_DONE\|^CRASH s37ab' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37ab' runs/alloc/status && exit 1
cd paper/strat/fusion
STRAT_REF=Q-bkl3_v5 STRAT_LABEL_SUFFIX=_v8 STRAT_WORKERS=8 timeout 7200 ../../../.venv/bin/python bw_valid.py Q-bkl3 > ../../../runs/strat/fusion/bw_valid_Q-bkl3_v8.log 2>&1
grep -q '^S37S_VALID' ../../../runs/strat/fusion/bw_valid_Q-bkl3_v8.log && echo "S37AB_VALID_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37ab valid $(date -Is)" >> ../../../runs/alloc/status
STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v8 timeout 1800 ../../../.venv/bin/python head_eval.py > ../../../runs/strat/fusion/head_eval_v8.log 2>&1
