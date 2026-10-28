#!/bin/bash
# doc 35 §38aa (cont.): combinations on top of X8axH (axis shift aimed at the J1 heads) - tri column search, joint fit.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while pgrep -f "[s]38aa_runner" > /dev/null; do sleep 20; done
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
run() { flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag $1 --psi psiF --qb qh50 --heads "$H" --kw "$2" --axis head >> $L/s38aa.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38aa.log; }
run HT '{"_steps":["shift","bw","kw","xp","tsnap","li"],"xp":{"sigma":0.15,"cap":0.10}}'
run HJ '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10}}'
run HJT '{"_steps":["shift","bw","kw","xp","tsnap","jgn","li"],"xp":{"sigma":0.15,"cap":0.10}}'
echo "S38AB_DONE $(date -Is)" >> $L/s38aa.log
