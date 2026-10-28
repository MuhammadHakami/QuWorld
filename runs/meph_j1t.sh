#!/bin/bash
# Fusion §41q J1T: J1 head + the §41f fixed-coordinate Thomson block (STRAT_THZ=1; never tried on the heads) vs J1M_s1
# (same machine), after the PCN / PCM psi trunks.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q PCN_DONE $L/PCN_s1.log 2>/dev/null; do sleep 60; done
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix \
  STRAT_THZ=1 STRAT_OUT_TAG=_J1T .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1T_s1.log 2>&1
echo "J1T_DONE $(date -Is)" >> $L/J1T_s1.log
