#!/bin/bash
# doc 35 §28 M1 addendum: exact_eps control (1%, 2%) on the same VALID instances and seeds, after the screen.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; P=runs/routefinder/s23/eal_md_vanilla_s1.pt
until grep -q S28_M1_SCREEN_DONE $S; do sleep 60; done
for seed in 1 2; do
  [ -f runs/qmethods/m1/m1_val_eal_md_vanilla_s1_s${seed}_ctrl.json ] && continue
  (cd baseline/routefinder && QW_GPU_GB=6 timeout 14400 ../../runs/capped.sh s28m1c 12 ../../.venv-vrp/bin/python -u \
     ../../paper/qmethods/run_mwis_vrp.py --split val --n 128 --policy ../../$P --seed $seed --selectors exact_eps1,exact_eps2 --tag _ctrl) \
     >> runs/qmethods/s28_m1.log 2>&1 || echo "CRASH s28 m1ctrl seed $seed $(date -Is)" >> $S
done; echo "S28_M1_CTRL_DONE $(date -Is)" >> $S
