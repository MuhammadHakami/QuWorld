#!/bin/bash
# Fusion §41q: J1 shape head seed 1 with Ip-sign canonical current inputs (canon.py) vs a same-machine control (J1 recipe,
# magfix inputs, s38p env). Screen: head_screen J1CN_s1 vs J1M_s1 (both mephiles); per-scalar R2 on reversed / normal shots.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix"
env $E STRAT_CANON=1 STRAT_OUT_TAG=_J1CN .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1CN_s1.log 2>&1
env $E STRAT_OUT_TAG=_J1M .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1M_s1.log 2>&1
echo "J1CN_DONE $(date -Is)" >> $L/J1CN_s1.log
