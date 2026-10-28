#!/bin/bash
# §45d: X-point GBDT heads (fit on .venv-tdc CatBoost GPU, eval on the main venv).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion/s45d_gbdt_xp.log
.venv-tdc/bin/python paper/strat/fusion/gbdt_xp.py fit >> $L 2>&1 || { echo "CRASH fit $(date -Is)" >> $L; exit 1; }
QW_GPU_GB=25 runs/capped.sh gbdtxp 30 .venv/bin/python paper/strat/fusion/gbdt_xp.py eval >> $L 2>&1 || { echo "CRASH eval $(date -Is)" >> $L; exit 1; }
echo "S45D_DONE $(date -Is)" >> $L
