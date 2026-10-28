#!/bin/bash
# TDC follow-ups (Sep 23): after the CheMeleon vanilla pass -> solubility vanilla retry (SMILES fix) -> MapLight all 22
# endpoints on GPU CatBoost (queued behind CheMeleon: both are small-kernel jobs that starve each other on a shared GPU).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=8
S=runs/tdc_chemeleon/status
until grep -q "PASS_DONE vanilla" "$S"; do sleep 120; done
[ -f runs/tdc_chemeleon/solubility_aqsoldb_vanilla.json ] || runs/capped.sh tdc-sol 12 .venv-tdc/bin/python paper/tdc/run_chemeleon.py solubility_aqsoldb vanilla \
  > runs/tdc_chemeleon/solubility_aqsoldb_vanilla.log 2>&1 && echo "result solubility_aqsoldb vanilla (retry): $(grep '\[chemeleon\]' runs/tdc_chemeleon/solubility_aqsoldb_vanilla.log | tail -1) $(date -Is)" >> "$S"
mkdir -p runs/tdc_maplight
runs/capped.sh maplight 12 .venv-tdc/bin/python paper/tdc/run_maplight.py all > runs/tdc_maplight/run.log 2>&1 \
  && echo "MAPLIGHT_DONE $(date -Is)" >> runs/tdc_maplight/status || echo "CRASH maplight $(date -Is)" >> runs/tdc_maplight/status
