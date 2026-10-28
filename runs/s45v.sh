#!/bin/bash
# §45s register label-free check (S45S_REG_ONLY) + §45v X-point tracking diagnostic, each under vt.lock (queued behind §45t / §45u).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
flock runs/strat/vt.lock env QW_GPU_GB=30 S45S_REG_ONLY=1 runs/capped.sh s45sfree 40 .venv/bin/python paper/strat/fusion/s45s_eval.py >> $L/s45s.log 2>&1 || echo "CRASH s45s regfree $(date -Is)" >> $L/s45s.log
echo "S45S_REGFREE_DONE $(date -Is)" >> $L/s45s.log
echo "[s45v] start $(date -Is)" >> $L/s45v.log
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45v 40 .venv/bin/python paper/strat/fusion/xp_track.py --n 50 >> $L/s45v.log 2>&1 || echo "CRASH s45v $(date -Is)" >> $L/s45v.log
echo "S45V_DONE $(date -Is)" >> $L/s45v.log
