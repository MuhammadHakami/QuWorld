#!/bin/bash
# Fusion §41q transfer check: HJl9 pipeline with ONE head (J1T_s1 = + fixed-coordinate Thomson vs J1M_s1 control, both mephiles)
# on the official VALID sample - how much of a +0.001 head R2 reaches the composite.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
for h in J1M J1T; do
  QW_GPU_GB=30 runs/capped.sh s41qg 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag HJ1$h --psi psiF --qb qh50 \
    --heads head_boundary_v8_${h}_s1 --kw "$KW" --axis head >> $L/s41q_gate2.log 2>&1
done
echo "S41Q_GATE2_DONE $(date -Is)" >> $L/s41q_gate2.log
