#!/bin/bash
# Fusion §41q mephiles chain (replaces meph_chain2.sh; QHT_s1 already running): after QHT -> QHCS_s1 (q/b heads with canonical
# currents + sign input; QHC lost on normal shots, won on reversed) -> J1MT_s1 -> J1CS_s1. Same-machine controls QHM_s1 / J1M_s1.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q -E "S37R_HEAD_DONE|Error" $L/QHT_s1.log 2>/dev/null; do sleep 60; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix"
env $E STRAT_CANON=2 STRAT_OUT_TAG=_QHCS .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHCS_s1.log 2>&1
env $E STRAT_SCAL_W=9.142857 STRAT_AUX_PSI=1 STRAT_OUT_TAG=_J1MT .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1MT_s1.log 2>&1
env $E STRAT_SCAL_W=9.142857 STRAT_CANON=2 STRAT_OUT_TAG=_J1CS .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1CS_s1.log 2>&1
echo "CHAIN3_DONE $(date -Is)" >> $L/J1CS_s1.log
