#!/bin/bash
# doc 35 §37s: after the v2 heads, head accuracy (head_eval) and the Q-bl3 pipeline with v2 heads on the sample.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37S_HEADS_DONE\|^CRASH s37s' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37s' runs/alloc/status && exit 1
cd paper/strat/fusion
STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v2 timeout 1800 ../../../.venv/bin/python head_eval.py > ../../../runs/strat/fusion/head_eval_v2.log 2>&1
STRAT_LABEL_SUFFIX=_v2 STRAT_WORKERS=8 timeout 7200 ../../../.venv/bin/python bw_valid.py Q-bl3 > ../../../runs/strat/fusion/bw_valid_Q-bl3_v2.log 2>&1
grep -q '^S37S_VALID' ../../../runs/strat/fusion/bw_valid_Q-bl3_v2.log && echo "S37S_VALID_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37s valid $(date -Is)" >> ../../../runs/alloc/status
