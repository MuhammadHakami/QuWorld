#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
until grep -q '^S38D_DONE' runs/alloc/status; do sleep 30; done
cd paper/strat/fusion && timeout 10800 ../../../.venv/bin/python s38d_eval.py > ../../../runs/strat/fusion/s38d_eval.log 2>&1
