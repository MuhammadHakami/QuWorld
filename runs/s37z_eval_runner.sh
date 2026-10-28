#!/bin/bash
# doc 35 §37z: head-accuracy gate after the 60-epoch heads.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37Z_HEADS_DONE\|^CRASH s37z' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37z' runs/alloc/status && exit 1
cd paper/strat/fusion && STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v6 timeout 1800 ../../../.venv/bin/python head_eval.py > ../../../runs/strat/fusion/head_eval_v6.log 2>&1
grep -q 'S37P_EVAL_DONE' ../../../runs/strat/fusion/head_eval_v6.log && echo "S37Z_EVAL_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37z eval $(date -Is)" >> ../../../runs/alloc/status
