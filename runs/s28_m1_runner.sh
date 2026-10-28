#!/bin/bash
# doc 35 §28 M1 screen: Rydberg MWIS move selection vs classical selectors, VALID (first 200), seeds 1-2,
# starting from the §23 EAL-fine-tuned vanilla policy (seed 1) as soon as it is saved.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; P=runs/routefinder/s23/eal_md_vanilla_s1.pt
until [ -f $P ]; do sleep 60; done; sleep 30
for seed in 1 2; do
  [ -f runs/qmethods/m1/m1_val_eal_md_vanilla_s1_s${seed}.json ] && continue
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 30; done
  (cd baseline/routefinder && QW_GPU_GB=6 timeout 14400 ../../runs/capped.sh s28m1 12 ../../.venv-vrp/bin/python -u \
     ../../paper/qmethods/run_mwis_vrp.py --split val --n 200 --policy ../../$P --seed $seed) >> runs/qmethods/s28_m1.log 2>&1 \
     || echo "CRASH s28 m1 seed $seed $(date -Is)" >> $S
done; echo "S28_M1_SCREEN_DONE $(date -Is)" >> $S
