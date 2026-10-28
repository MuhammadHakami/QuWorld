#!/bin/bash
# Fusion §41q J1MT (mephiles, after the QHC chain): J1 head + the 64 psi PCA coefficients as auxiliary targets (STRAT_AUX_PSI=1,
# weight 1 per coefficient as the rays) - heads are data-limited (learning-curve exponent ~0.6), the coefficients add dense
# supervision per frame. vs J1M_s1 (same machine). Prediction / scoring use the 71 main outputs only.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q QHC_DONE $L/QHC_s1.log 2>/dev/null; do sleep 60; done
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix \
  STRAT_AUX_PSI=1 STRAT_OUT_TAG=_J1MT .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1MT_s1.log 2>&1
echo "J1MT_DONE $(date -Is)" >> $L/J1MT_s1.log
