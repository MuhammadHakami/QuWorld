#!/bin/bash
# Doc 35 §16 pilot: routed node register ladder. Usage: runs/rf_route_runner.sh <lane 0-1>
cd /home/mohhakami/Documents/paper/quworld/baseline/routefinder; lane=$1; i=0; S=../../runs/routefinder/status; out=../../runs/routefinder/routed
for v in route5 route5-mlp route10 route10-mlp route20 route20-mlp route30 route30-mlp; do
  i=$((i+1)); [ $((i % 2)) -ne $lane ] && continue; [ -f $out/eal_md_${v}_s1.json ] && continue
  timeout 3600 ../../runs/capped.sh rfr-$lane 24 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $v --setting md --seed 1 --epochs 1 --train-size 2000 --precision 32 --out $out > $out/$v.log 2>&1 \
    && echo "RF_ROUTE_DONE $v $(date -Is)" >> $S || echo "CRASH rf-route $v $(date -Is)" >> $S
done
echo "RF_ROUTE_LANE_DONE $lane $(date -Is)" >> $S
