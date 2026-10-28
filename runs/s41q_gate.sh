#!/bin/bash
# Fusion §41q gate: HJl9 with psi from the canonical PSI-F trunks (PCN s2 / s1, mephiles, TRAIN only) on the official VALID sample
# vs HJl9 0.99562 (same heads / qb / post-processing). Runs beside the final-build training (no GPU lock; small).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
QW_GPU_GB=30 runs/capped.sh s41qg 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag HJl9C --psi psiFC --qb qh50 --heads "$H" --kw "$KW" --axis head >> $L/s41q_gate.log 2>&1
echo "S41Q_GATE_DONE $(date -Is)" >> $L/s41q_gate.log
