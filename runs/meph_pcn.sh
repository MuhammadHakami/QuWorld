#!/bin/bash
# Fusion §41q: PSI-F trunk config with Ip-sign canonical current inputs (cfg canon) vs control, seeds 1 and 2, same machine
# (mephiles), after the J1CN / J1M heads. Read-out: quick VALID r2_psi + err_anatomy on reversed vs normal shots.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1CN_DONE $L/J1CN_s1.log 2>/dev/null; do sleep 60; done
# §41q J1H: J1 head + full-rate history inputs (hist_full.py, STRAT_HFULL=1) vs J1M_s1 (same machine), before the psi trunks
env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix \
  STRAT_HFULL=1 STRAT_OUT_TAG=_J1H .venv-vrp/bin/python paper/strat/fusion/head_boundary.py 1 > $L/J1H_s1.log 2>&1
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"'
for s in 1 2; do
  for arm in PCN PCM; do
    C=$([ $arm = PCN ] && echo ', "canon": true' || echo '')
    .venv-vrp/bin/python paper/sophelio/sota_train.py --tag ${arm}_s$s --score-val 0 --cfg "{$CFG, \"seed\": $s$C}" > $L/${arm}_s$s.log 2>&1
  done
done
echo "PCN_DONE $(date -Is)" >> $L/PCN_s1.log
