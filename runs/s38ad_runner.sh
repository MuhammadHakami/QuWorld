#!/bin/bash
# doc 35 §38aa (cont.): tri column search with less boundary movement (D_LCFS cost of HT), on X8axH.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while pgrep -f "[s]38ac_runner" > /dev/null; do sleep 20; done
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
run() { flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag $1 --psi psiF --qb qh50 --heads "$H" --kw "$2" --axis head >> $L/s38aa.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38aa.log; }
run HTh '{"_steps":["shift","bw","kw","xp","tsnap","li"],"xp":{"sigma":0.15,"cap":0.10},"tsnap":{"shifts":[-0.5,0.5]}}'
run HTn '{"_steps":["shift","bw","kw","xp","tsnap","li"],"xp":{"sigma":0.15,"cap":0.10},"tsnap":{"sigma":0.08}}'
echo "S38AD_DONE $(date -Is)" >> $L/s38aa.log
