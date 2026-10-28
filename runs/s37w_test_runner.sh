#!/bin/bash
# doc 35 §37w: TEST once (Q-bkl3 with the v5 heads; base Q-bl3-v5), after the §37v TEST has written bw_test_v5.json.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until [ -f runs/strat/fusion/bw_test_v5.json ]; do sleep 30; done
cd paper/strat/fusion && STRAT_CUDA="" STRAT_LABEL_SUFFIX=_v5 timeout 10800 ../../../.venv/bin/python bw_test.py Q-bkl3 > ../../../runs/strat/fusion/bw_test_kw_v5.log 2>&1
grep -q '^S37Q_TEST' ../../../runs/strat/fusion/bw_test_kw_v5.log && echo "S37W_TEST_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37w test $(date -Is)" >> ../../../runs/alloc/status
