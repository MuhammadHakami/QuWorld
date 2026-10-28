#!/bin/bash
# Fusion §41q mephiles chain (replaces meph_qhc.sh / meph_j1mt.sh; QHM_s1 already running): after QHM -> QHC_s1 (canonical
# currents) -> QHT_s1 (+ fixed-coordinate Thomson; J1T gained ~10 sd on the shape heads) -> J1MT_s1 (psi coefficients as aux
# targets) -> J1CS_s1 (canonical + sign input). q/b arms vs QHM_s1, shape arms vs J1M_s1 (same machine).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q -E "S37R_HEAD_DONE|Error" $L/QHM_s1.log 2>/dev/null; do sleep 60; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix"
env $E STRAT_CANON=1 STRAT_OUT_TAG=_QHC .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHC_s1.log 2>&1
env $E STRAT_THZ=1 STRAT_OUT_TAG=_QHT .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 qb > $L/QHT_s1.log 2>&1
env $E STRAT_SCAL_W=9.142857 STRAT_AUX_PSI=1 STRAT_OUT_TAG=_J1MT .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1MT_s1.log 2>&1
env $E STRAT_SCAL_W=9.142857 STRAT_CANON=2 STRAT_OUT_TAG=_J1CS .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1CS_s1.log 2>&1
echo "CHAIN2_DONE $(date -Is)" >> $L/J1CS_s1.log
