#!/bin/bash
# doc 35 §37aa: head-accuracy gate after the scalar-weighted heads.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37AA_HEADS_DONE\|^CRASH s37aa' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37aa' runs/alloc/status && exit 1
cd paper/strat/fusion && STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v7 timeout 1800 ../../../.venv/bin/python head_eval.py > ../../../runs/strat/fusion/head_eval_v7.log 2>&1
grep -q 'S37P_EVAL_DONE' ../../../runs/strat/fusion/head_eval_v7.log && echo "S37AA_EVAL_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37aa eval $(date -Is)" >> ../../../runs/alloc/status
