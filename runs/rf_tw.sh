#!/bin/bash
# mephiles (routing lane, doc 35 §39c): bf16 EAL-MD §39d time-window-gated routed hybrid (route30-tw), seeds 1-5,
# paired against classical bf16 (8.654) and route30 (8.614).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; O=runs/routefinder/bf16_mephiles; mkdir -p $O
for seed in 1 2 3 4 5; do for v in route30-tw; do
  [ -f $O/eal_md_${v}_s$seed.json ] && continue
  (cd baseline/routefinder && systemd-run --user --scope --quiet -p MemoryMax=22G -p MemorySwapMax=0 timeout 36000 \
    ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $v --setting md --seed $seed --epochs 10 --train-size 10000 --out ../../$O) \
    >> $O/runner_tw.log 2>&1 || echo "CRASH $v s$seed $(date -Is)" >> $O/runner_tw.log
done; done
echo "RF_TW_DONE $(date -Is)" >> $O/runner_tw.log
