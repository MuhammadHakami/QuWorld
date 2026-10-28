#!/bin/bash
# §45s X-point / label-free read-out (s45s_eval.py) after the twins; the register VT router read moved to s45s_reg.sh.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45s.log
until grep -q "^S45S_TRAINED" $LOG; do sleep 120; done
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45sev 40 .venv/bin/python paper/strat/fusion/s45s_eval.py >> $LOG 2>&1 || echo "CRASH s45s eval $(date -Is)" >> $LOG
echo "S45S_DONE $(date -Is)" >> $LOG
