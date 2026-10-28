#!/bin/bash
# §42t: continuous X-point oracle on the §42q 60-shot VT sample (diagnostic).
cd /home/mohhakami/Documents/paper/quworld/paper/strat/fusion; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 QW_GPU_GB=20; LOG=../../../runs/strat/fusion/xpo.log
../../../.venv/bin/python xp_oracle.py --n 60 >> $LOG 2>&1 || echo "CRASH xpo $(date -Is)" >> $LOG
echo "S42T_DONE $(date -Is)" >> $LOG
