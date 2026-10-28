#!/bin/bash
# doc 35 §37ab: TEST once (Q-bkl3 with the v8 heads vs Q-bkl3 with the v5 heads), after the §37y TEST.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37Y_TEST_DONE\|^CRASH s37y test' runs/alloc/status; do sleep 30; done
cd paper/strat/fusion && STRAT_CUDA="" STRAT_BASE_TAG=_v5 STRAT_LABEL_SUFFIX=_v8 timeout 10800 ../../../.venv/bin/python bw_test.py Q-bkl3 > ../../../runs/strat/fusion/bw_test_kw_v8.log 2>&1
grep -q '^S37Q_TEST' ../../../runs/strat/fusion/bw_test_kw_v8.log && echo "S37AB_TEST_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37ab test $(date -Is)" >> ../../../runs/alloc/status
