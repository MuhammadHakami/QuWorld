#!/bin/bash
# Doc 35 §14: RouteFinder qubit ladder pilot. Usage: runs/rf_scaling_runner.sh <lane 0-2>
cd /home/mohhakami/Documents/paper/quworld/baseline/routefinder; lane=$1; i=0; S=../../runs/routefinder/status
for v in vanilla q5-part=1x5 q5-part=1x5-mlp q10-part=1x10 q10-part=1x10-mlp q20-part=2x10 q20-part=2x10-mlp q30 q30-mlp; do
  i=$((i+1)); [ $((i % 3)) -ne $lane ] && continue
  out=../../runs/routefinder/scaling; mkdir -p $out; [ -f $out/eal_md_${v}_s1.json ] && continue
  timeout 2400 ../../runs/capped.sh rf-$lane 24 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $v --setting md --seed 1 --epochs 1 --train-size 2000 --precision 32 --out $out > $out/$v.log 2>&1 \
    && echo "RF_SCALE_DONE $v $(date -Is)" >> $S || echo "CRASH rf-scale $v $(date -Is)" >> $S
done
echo "RF_SCALE_LANE_DONE $lane $(date -Is)" >> $S
