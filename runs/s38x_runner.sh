#!/bin/bash
# §38x combined pipeline (registered 2026-10-03 07:30 +03, before any result of this combination):
# PSI-F psi (3 seeds) + QF q/b 50/50 with entry + 6-seed corrected-input heads (G1_s1 + H1F_s2..6) + X3 warp chain.
# Rule: VALID sample (official parts, paired shot bootstrap vs frozen base) CI lower bound > 0 -> TEST once;
# TEST S > 0.993691 (current best local TEST) -> public build. Upload decided by hand against the public 0.99353.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q "^S38E6_DONE" $L/s38e6.log 2>/dev/null; do sleep 60; done
H=$(grep "^S38E6_DONE" $L/s38e6.log | awk '{print $2}')
KW='{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10}}'
A=(--psi psiF --qb qf50 --heads "$H" --kw "$KW")
echo "S38X heads $H $(date -Is)" > $L/s38x.log
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag X6 "${A[@]}" >> $L/s38x.log 2>&1
LO=$(grep "GPU_SAMPLE X6 official" $L/s38x.log | sed 's/.*\[\([-+0-9.]*\),.*/\1/')
echo "S38X sample official lo ${LO:-NA}" >> $L/s38x.log
python3 -c "import sys; sys.exit(0 if float('${LO:-0}') > 0 else 1)" || { echo "S38X_DONE stop: sample CI not above 0" >> $L/s38x.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split test --tag X6 "${A[@]}" >> $L/s38x.log 2>&1
T=$(grep "^GPU_TEST X6" $L/s38x.log | sed 's/.*S \([0-9.]*\).*/\1/')
python3 -c "import sys; sys.exit(0 if float('${T:-0}') > 0.993691 else 1)" || { echo "S38X_DONE stop: TEST ${T:-NA} not above 0.993691" >> $L/s38x.log; exit 0; }
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag X6 "${A[@]}" >> $L/s38x.log 2>&1
echo "S38X_DONE TEST $T public built $(date -Is)" >> $L/s38x.log
