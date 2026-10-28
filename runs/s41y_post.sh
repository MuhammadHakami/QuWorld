#!/bin/bash
# Fusion §41y (registered 2026-10-11, before any result): post-processing transfer fixes, inference only. Transfer check on the F2
# public psi: li - 0.1% of frames (heads li 2.6-4.8, shot ends) carry 97% of the li transfer error (li warp beta capped at 0.9); tri
# transfer residual 5.7-6.2% of the tri spread across most shots. Arms on the 180-shot VALID sample (official scorer), HJl9 components
# (psiF, J1 x6, QH x6, qh50, head axis): B = current; L = li cap 2.5 (cap_lo -0.9, the profile map stays monotone to 3);
# X = xp n_it 6, dk 0.006; LX = both. Rule: adopt a change if its official S beats B by >= +0.00005 with consistency up and D_LCFS not worse.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion/s41y_post.log
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
for arm in 'B|{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' \
           'L|{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":2.5,"cap_lo":-0.9}}' \
           'X|{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10,"n_it":6,"dk":0.006},"li":{"cap":0.9}}' \
           'LX|{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10,"n_it":6,"dk":0.006},"li":{"cap":2.5,"cap_lo":-0.9}}'; do
  n=${arm%%|*}; KW=${arm#*|}
  QW_GPU_GB=24 runs/capped.sh s41y 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag PP$n --psi psiF --qb qh50 --qbheads "$Q" \
    --heads "$J" --kw "$KW" --axis head >> $L 2>&1 || echo "CRASH $n" >> $L
done
echo "S41Y_DONE $(date -Is)" >> $L
