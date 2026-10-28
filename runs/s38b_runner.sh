#!/bin/bash
# doc 35 §38b: CV specialists screen, one arm at a time through the GPU lock.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
S=runs/alloc/status; L=runs/strat/fusion/cv_trunk.log
run() { [ -f runs/strat/fusion/cv_trunk/$1.json ] && return; flock runs/strat/gpu.lock env QW_GPU_GB=24 timeout 5400 runs/capped.sh cvtrunk 48 .venv/bin/python paper/strat/fusion/cv_trunk.py "$1" "$2" >> $L 2>&1 || echo "CRASH s38b $1 $(date -Is)" >> $S; }
run P150C '{"epochs": 150, "clip": 1.0}'
run P90C0 '{"epochs": 90, "clip": 1.0, "w_scal": 0.0}'
run Q4E90 '{"depth": 4, "epochs": 90}'
run Q4W3 '{"depth": 4, "w_scal": 4.5}'
echo "S38B_DONE $(date -Is)" >> $S
