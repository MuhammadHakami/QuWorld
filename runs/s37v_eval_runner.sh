#!/bin/bash
# doc 35 §37v: after the v5 heads (and the §37u sample score), head accuracy and Q-bl3 with v4 heads on the sample.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37V_HEADS_DONE\|^CRASH s37v' runs/alloc/status && grep -q '^S37U_VALID_DONE\|^CRASH s37u valid' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37v' runs/alloc/status && exit 1
cd paper/strat/fusion
STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v5 timeout 1800 ../../../.venv/bin/python head_eval.py > ../../../runs/strat/fusion/head_eval_v5.log 2>&1
STRAT_LABEL_SUFFIX=_v5 STRAT_WORKERS=10 timeout 7200 ../../../.venv/bin/python bw_valid.py Q-bl3 > ../../../runs/strat/fusion/bw_valid_Q-bl3_v5.log 2>&1
grep -q '^S37S_VALID' ../../../runs/strat/fusion/bw_valid_Q-bl3_v5.log && echo "S37V_VALID_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37v valid $(date -Is)" >> ../../../runs/alloc/status
