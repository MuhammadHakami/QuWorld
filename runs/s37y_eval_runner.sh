#!/bin/bash
# doc 35 §37y: Q-bkl3 (v5 heads) with the 6-seed head mean vs the 3-seed mean on the sample, after seeds 4-6.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37Y_HEADS_DONE\|^CRASH s37y' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37y' runs/alloc/status && exit 1
cd paper/strat/fusion
STRAT_HEAD_SEEDS=1,2,3,4,5,6 STRAT_LABEL_SUFFIX=_v5 STRAT_WORKERS=8 timeout 7200 ../../../.venv/bin/python bw_valid.py Q-bkl3 > ../../../runs/strat/fusion/bw_valid_Q-bkl3_v5_6s.log 2>&1
grep -q '^S37S_VALID' ../../../runs/strat/fusion/bw_valid_Q-bkl3_v5_6s.log && echo "S37Y_VALID_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37y valid $(date -Is)" >> ../../../runs/alloc/status
