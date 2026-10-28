#!/bin/bash
# doc 35 §38aa: joint Gauss-Newton shape fit (jgn) and head-axis target on the VALID sample (official composite vs X8).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
run() { flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag $1 --psi psiF --qb qh50 --heads "$H" --kw "$2" ${3:+--axis $3} >> $L/s38aa.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38aa.log; }
run X8axH '{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}' head
run X8J '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10}}'
run J '{"_steps":["shift","bw","jgn","li"]}'
echo "S38AA_DONE $(date -Is)" >> $L/s38aa.log
