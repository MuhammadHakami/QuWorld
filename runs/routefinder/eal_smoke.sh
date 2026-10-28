#!/bin/bash
# One-epoch smoke of the RouteFinder EAL arms harness (register arm, setting md); writes EAL_SMOKE_OK on success.
cd /home/mohhakami/Documents/paper/quworld/baseline/routefinder; export PYTHONUNBUFFERED=1
../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py mr30ryd --setting md --seed 1 --epochs 1 --train-size 1000 \
  --out ../../runs/routefinder/smoke > ../../runs/routefinder/eal_smoke.log 2>&1 \
  && [ -f ../../runs/routefinder/smoke/eal_md_mr30ryd_s1.json ] && touch ../../runs/routefinder/EAL_SMOKE_OK
echo "smoke exit $? $(date -Is)" >> ../../runs/routefinder/eal_smoke.log
