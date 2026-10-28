#!/bin/bash
# Fusion §42m (replaces the §42l runner): PQ2 and the PQ2 + LS_ctrl mix with the axis guard (fallback psiF, 0.25 m), VB / ns chains, all VT shots.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42m_vt.log
until grep -q "^S42K_DONE" $L/s42k.log 2>/dev/null; do sleep 30; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
X='"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}'
KWS="{\"VB\":{\"_steps\":[\"shift\",\"bw\",\"kw\",\"xp\",\"jgn\",\"li\"],$X},\"ns\":{\"_steps\":[\"bw\",\"kw\",\"xp\",\"jgn\",\"li\"],$X}}"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42mvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag PG \
  --psi psiF --psis "PQ2~g,PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$J" --kws "$KWS" --axis head >> $LOG 2>&1 || echo "CRASH vt PG $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py LS_ctrl_VB "PQ2~g_VB" "PQ2~g_ns" "PQmix~g_VB" "PQmix~g_ns" PQ2_VB >> ../../../$LOG 2>&1
echo "S42M_VT_DONE $(date -Is)" >> ../../../$LOG
