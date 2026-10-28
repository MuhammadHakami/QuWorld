#!/bin/bash
# doc 35 §37y: TEST once (Q-bkl3 v5, 6-seed heads vs 3-seed), after the §37w TEST.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until grep -q '^S37W_TEST_DONE\|^CRASH s37w' runs/alloc/status; do sleep 30; done
grep -q '^CRASH s37w' runs/alloc/status && exit 1
cd paper/strat/fusion && STRAT_CUDA="" STRAT_HEAD_SEEDS=1,2,3,4,5,6 STRAT_LABEL_SUFFIX=_v5 timeout 10800 ../../../.venv/bin/python bw_test.py Q-bkl3 > ../../../runs/strat/fusion/bw_test_kw_v5_6s.log 2>&1
grep -q '^S37Q_TEST' ../../../runs/strat/fusion/bw_test_kw_v5_6s.log && echo "S37Y_TEST_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37y test $(date -Is)" >> ../../../runs/alloc/status
