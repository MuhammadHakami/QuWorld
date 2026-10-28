#!/bin/bash
# Doc 35 §15: RouteFinder full-protocol qubit ladder, 3 lanes, after the §14 pilot. Usage: runs/rf_full_runner.sh <lane 0-2>
cd /home/mohhakami/Documents/paper/quworld; S=runs/routefinder/status; lane=$1; i=0
while [ "$(grep -c RF_SCALE_LANE_DONE $S)" -lt 3 ] || [ "$(grep -c RF_ROUTE_LANE_DONE $S)" -lt 2 ]; do sleep 60; done
cd baseline/routefinder; out=../../runs/routefinder/full; mkdir -p $out
for spec in "vanilla 1" "q30 1" "q30-mlp 1" "route30 1" "route30-mlp 1" "route10 1" "route10-mlp 1" "q5-part=1x5 1" "q5-part=1x5-mlp 1" "q10-part=1x10 1" "q10-part=1x10-mlp 1" "q20-part=2x10 1" "q20-part=2x10-mlp 1" "vanilla 2" "q30 2" "q30-mlp 2" "vanilla 3" "q30 3" "q30-mlp 3"; do
  set -- $spec; i=$((i+1)); [ $((i % 3)) -ne $lane ] && continue
  [ -f $out/eal_md_${1}_s$2.json ] && continue
  timeout 28800 ../../runs/capped.sh rff-$lane 24 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $1 --setting md --seed $2 --epochs 10 --train-size 10000 --precision 32 --out $out > $out/${1}_s$2.log 2>&1 \
    && echo "RF_FULL_DONE $1 s$2 $(date -Is)" >> ../../$S || echo "CRASH rf-full $1 s$2 $(date -Is)" >> ../../$S
done
echo "RF_FULL_LANE_DONE $lane $(date -Is)" >> ../../$S
