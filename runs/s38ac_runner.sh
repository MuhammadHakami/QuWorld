#!/bin/bash
# doc 35 §38aa (cont.): keep the joint fit's kappa gain without its li / D_LCFS losses (on X8axH = head-aimed axis).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while pgrep -f "[s]38ab_runner" > /dev/null; do sleep 20; done
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
run() { flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag $1 --psi psiF --qb qh50 --heads "$H" --kw "$2" --axis head >> $L/s38aa.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38aa.log; }
run HJl9 '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
run HJnv '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"jgn":{"fit":[true,true,true,true,true,false]}}'
echo "S38AC_DONE $(date -Is)" >> $L/s38aa.log
