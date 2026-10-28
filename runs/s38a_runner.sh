#!/bin/bash
# doc 35 §38a: CV trunk screen, one arm at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
S=runs/alloc/status; L=runs/strat/fusion/cv_trunk.log
run() { [ -f runs/strat/fusion/cv_trunk/$1.json ] && return; flock runs/strat/gpu.lock env QW_GPU_GB=24 timeout 3600 runs/capped.sh cvtrunk 48 .venv/bin/python paper/strat/fusion/cv_trunk.py "$1" "$2" >> $L 2>&1 || echo "CRASH s38a $1 $(date -Is)" >> $S; }
run E90 '{"epochs": 90}'
run W4096 '{"width": 4096}'
run D4 '{"depth": 4}'
run E90C '{"epochs": 90, "clip": 1.0}'
echo "S38A_DONE $(date -Is)" >> $S
