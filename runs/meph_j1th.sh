#!/bin/bash
# Fusion §41q full-size confirmation of the proxy winner thz + hfull (proxy +0.00034, 4/5 seeds vs thz): J1TH_s1 = J1 recipe + thz + 24
# full-rate history inputs vs J1T_s1 (mephiles, same machine). Runs beside the small proxy jobs.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix \
  STRAT_THZ=1 STRAT_HFULL=1 STRAT_OUT_TAG=_J1TH .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1TH_s1.log 2>&1
echo "J1TH_DONE $(date -Is)" >> $L/J1TH_s1.log
