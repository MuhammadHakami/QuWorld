#!/bin/bash
# After F8's build: F8K = post_kw2 on F8's public psi (the §42y post-pass; uploaded only if F7KC beats F7C on public).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f8k.log
until grep -q "^F8_BUILT\|^F8_FAIL" $L/f8.log 2>/dev/null; do sleep 30; done
grep -q "^F8_BUILT" $L/f8.log || { echo "F8K_SKIP $(date -Is)" >> $LOG; exit 0; }
QW_GPU_GB=12 runs/capped.sh kw2f8 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F8 --tag F8K >> $LOG 2>&1 && echo "F8K_DONE $(date -Is)" >> $LOG || echo "CRASH F8K $(date -Is)" >> $LOG
