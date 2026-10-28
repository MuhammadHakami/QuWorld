#!/bin/bash
# doc 35 §37q: TEST once for the picked arm, after the §37o TEST has written li_test.json.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
until [ -f runs/strat/fusion/li_test.json ]; do sleep 30; done
cd paper/strat/fusion && STRAT_CUDA="" timeout 7200 ../../../.venv/bin/python bw_test.py Q-bl3 > ../../../runs/strat/fusion/bw_test.log 2>&1
grep -q '^S37Q_TEST' ../../../runs/strat/fusion/bw_test.log && echo "S37Q_TEST_DONE $(date -Is)" >> ../../../runs/alloc/status || echo "CRASH s37q test $(date -Is)" >> ../../../runs/alloc/status
