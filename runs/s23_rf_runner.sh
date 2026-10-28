#!/bin/bash
# doc 35 §23: RouteFinder EAL on unseen multi-depot variants (their protocol: eal-full, lr 3e-4, 10 epochs x 10k, batch 128,
# last epoch tested on every unseen MD variant), seed 1, one run at a time; policies kept for the route figure.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; O=runs/routefinder/s23; mkdir -p $O
until grep -q S23_ARENA_DONE $S; do sleep 60; done
for v in vanilla q30-geo=3d q30-geo=3d-mlp; do
  [ -f $O/eal_md_${v}_s1.json ] && continue
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 40 ]; do sleep 60; done
  (cd baseline/routefinder && QW_GPU_GB=24 timeout 28800 ../../runs/capped.sh s23rf 60 ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py \
     "$v" --setting md --seed 1 --out ../../$O) >> runs/s23_rf.log 2>&1 || echo "CRASH s23 rf $v $(date -Is)" >> $S
done; echo "S23_RF_DONE $(date -Is)" >> $S
