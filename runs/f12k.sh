#!/bin/bash
# After F12's build: F12K = post_kw2 on F12 (kw2 adopted from F7KC).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f12k.log
until grep -q "^F12_BUILT\|^F12_FAIL\|^CRASH" $L/s43g.log 2>/dev/null; do sleep 60; done
grep -q "^F12_BUILT" $L/s43g.log || { echo "F12K_SKIP $(date -Is)" >> $LOG; exit 0; }
QW_GPU_GB=12 runs/capped.sh kw2f12 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F12 --tag F12K \
  --heads "$(grep -o 'heads head_boundary[^ ]*' $L/s43g.log | head -1 | cut -d' ' -f2)" >> $LOG 2>&1 && echo "F12K_DONE $(date -Is)" >> $LOG || echo "CRASH F12K $(date -Is)" >> $LOG
