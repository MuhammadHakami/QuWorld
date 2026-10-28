#!/bin/bash
# doc 35 §38p: J1 heads (4096x4, magfix) seeds 2-6, then X7 = X6 pipeline with the 6-seed J1 heads: sample -> TEST -> public.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
for seed in 2 3 4 5 6; do
  [ -f $L/head_boundary_v8_J1_s$seed.npz ] && continue
  flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 \
    STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_OUT_TAG=_J1 QW_GPU_GB=40 runs/capped.sh s38p 40 .venv/bin/python paper/strat/fusion/head_boundary.py $seed \
    >> $L/s38p.log 2>&1 || echo "CRASH J1 s$seed $(date -Is)" >> $L/s38p.log
done
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
.venv/bin/python paper/strat/fusion/head_screen.py "$H" "head_boundary_v8_G1_s1+head_boundary_v8_H1F_s2+head_boundary_v8_H1F_s3+head_boundary_v8_H1F_s4+head_boundary_v8_H1F_s5+head_boundary_v8_H1F_s6" >> $L/s38p.log 2>&1
KW='{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}'; A=(--psi psiF --qb qf50 --heads "$H" --kw "$KW")
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag X7 "${A[@]}" >> $L/s38p.log 2>&1
S=$(grep "GPU_SAMPLE X7 official" $L/s38p.log | sed 's/.*: S \([0-9.]*\).*/\1/'); LO=$(grep "GPU_SAMPLE X7 official" $L/s38p.log | sed 's/.*\[\([-+0-9.]*\),.*/\1/')
python3 -c "import sys; sys.exit(0 if float('${LO:-0}') > 0 and float('${S:-0}') > 0.99511 else 1)" || { echo "S38P_DONE stop: sample S ${S:-NA} lo ${LO:-NA}" >> $L/s38p.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split test --tag X7 "${A[@]}" >> $L/s38p.log 2>&1
T=$(grep "^GPU_TEST X7" $L/s38p.log | sed 's/.*S \([0-9.]*\).*/\1/')
python3 -c "import sys; sys.exit(0 if float('${T:-0}') > 0.994904 else 1)" || { echo "S38P_DONE stop: TEST ${T:-NA} not above 0.994904" >> $L/s38p.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag X7 "${A[@]}" >> $L/s38p.log 2>&1
echo "S38P_DONE TEST $T public built $(date -Is)" >> $L/s38p.log
