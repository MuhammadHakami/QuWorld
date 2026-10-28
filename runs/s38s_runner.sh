#!/bin/bash
# doc 35 §38s: QH q/b heads (4096x4, magfix) seeds 2-6, then X8 = X7 + q/b from 6-seed QH 50/50 with the entry.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
for seed in 2 3 4 5 6; do
  [ -f $L/head_qb_QH_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 \
    STRAT_INPUTS=magfix STRAT_OUT_TAG=_QH QW_GPU_GB=40 runs/capped.sh s38s 40 .venv/bin/python paper/strat/fusion/head_boundary.py $seed qb \
    >> $L/s38s.log 2>&1 || echo "CRASH QH s$seed $(date -Is)" >> $L/s38s.log
done
QH=$(for s in 1 2 3 4 5 6; do printf "head:head_qb_QH_s$s+"; done | sed 's/+$//')
STRAT_TAG=_s38s flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/qb_eval.py "$QH@entry" >> $L/s38s.log 2>&1
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}'; A=(--psi psiF --qb qh50 --heads "$H" --kw "$KW")
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag X8 "${A[@]}" >> $L/s38s.log 2>&1
S=$(grep "GPU_SAMPLE X8 official" $L/s38s.log | sed 's/.*: S \([0-9.]*\).*/\1/'); LO=$(grep "GPU_SAMPLE X8 official" $L/s38s.log | sed 's/.*\[\([-+0-9.]*\),.*/\1/')
python3 -c "import sys; sys.exit(0 if float('${LO:-0}') > 0 and float('${S:-0}') > 0.99512 else 1)" || { echo "S38S_DONE stop: sample S ${S:-NA} lo ${LO:-NA}" >> $L/s38s.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split test --tag X8 "${A[@]}" >> $L/s38s.log 2>&1
T=$(grep "^GPU_TEST X8" $L/s38s.log | sed 's/.*S \([0-9.]*\).*/\1/')
python3 -c "import sys; sys.exit(0 if float('${T:-0}') > 0.994934 else 1)" || { echo "S38S_DONE stop: TEST ${T:-NA} not above 0.994934" >> $L/s38s.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag X8 "${A[@]}" >> $L/s38s.log 2>&1
echo "S38S_DONE TEST $T public built $(date -Is)" >> $L/s38s.log
