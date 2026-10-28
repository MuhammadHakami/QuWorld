#!/bin/bash
# doc 35 §37t: after the v3 heads (and the §37s sample score), head accuracy and Q-bl3 with v3 heads on the sample.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37T_HEADS_DONE\|^CRASH s37t' runs/alloc/status && grep -q '^S37S_VALID_DONE\|^CRASH s37s' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37t' runs/alloc/status && exit 1
cd paper/strat/fusion
STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v3 timeout 1800 ../../../.venv/bin/python head_eval.py > ../../../runs/strat/fusion/head_eval_v3.log 2>&1
STRAT_LABEL_SUFFIX=_v3 STRAT_WORKERS=10 timeout 7200 ../../../.venv/bin/python bw_valid.py Q-bl3 > ../../../runs/strat/fusion/bw_valid_Q-bl3_v3.log 2>&1
grep -q '^S37S_VALID' ../../../runs/strat/fusion/bw_valid_Q-bl3_v3.log && echo "S37T_VALID_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37t valid $(date -Is)" >> ../../../runs/alloc/status
