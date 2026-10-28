#!/bin/bash
# Fusion §41q (mephiles, after J1T): canonical inputs for the q/b heads (q95 ~ 1/|Ip| is kinked in raw currents; psi trunks gained
# 34-51%) - QHC_s1 (STRAT_CANON=1) vs same-machine control QHM_s1 - then J1CS_s1 (J1 + canonical currents + sign input) vs J1M_s1.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1T_DONE $L/J1T_s1.log 2>/dev/null; do sleep 60; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix"
env $E STRAT_OUT_TAG=_QHM .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHM_s1.log 2>&1
env $E STRAT_CANON=1 STRAT_OUT_TAG=_QHC .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHC_s1.log 2>&1
env $E STRAT_SCAL_W=9.142857 STRAT_CANON=2 STRAT_OUT_TAG=_J1CS .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1CS_s1.log 2>&1
echo "QHC_DONE $(date -Is)" >> $L/QHC_s1.log
