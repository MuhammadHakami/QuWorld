#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
until grep -q '^S38C_DONE' runs/alloc/status; do sleep 30; done
cd paper/strat/fusion && timeout 7200 ../../../.venv/bin/python s38c_eval.py > ../../../runs/strat/fusion/s38c_eval.log 2>&1
