#!/bin/bash
# doc 35 §38aa: HJl9 (head-aimed axis + joint Gauss-Newton shape fit + li cap 0.9) - TEST once, gate vs X8 TEST 0.994967, public build.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'; A=(--psi psiF --qb qh50 --heads "$H" --kw "$KW" --axis head)
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split test --tag HJl9 "${A[@]}" >> $L/s38ae.log 2>&1
T=$(grep "^GPU_TEST HJl9" $L/s38ae.log | sed 's/.*S \([0-9.]*\).*/\1/')
python3 -c "import sys; sys.exit(0 if float('${T:-0}') > 0.994967 else 1)" || { echo "S38AE_DONE stop: TEST ${T:-NA} not above 0.994967" >> $L/s38ae.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag HJl9 "${A[@]}" >> $L/s38ae.log 2>&1
echo "S38AE_DONE TEST $T public built $(date -Is)" >> $L/s38ae.log
