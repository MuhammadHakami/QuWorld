#!/bin/bash
# doc 35 §41f: full fixed-coordinate Thomson extraction (train cache rows, then public-test rows), 8 CPU workers.
cd /home/mohhakami/Documents/paper/quworld/paper/strat/fusion; export PYTHONUNBUFFERED=1
../../../.venv/bin/python thz_feats.py --all > ../../../runs/strat/fusion/thz_all.log 2>&1
../../../.venv/bin/python thz_feats.py --public > ../../../runs/strat/fusion/thz_public.log 2>&1
echo THZ_EXTRACT_DONE >> ../../../runs/strat/fusion/thz_all.log
