#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42l.log
until grep -q "^S42K_DONE" $L/s42k.log 2>/dev/null; do sleep 60; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS='{"ns":{"_steps":["bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}},"nj":{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42lvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag PL \
  --psi psiF --psis PQ2 --qb qh50 --qbheads "$Q" --heads "$J" --kws "$KWS" --axis head >> $LOG 2>&1 || echo "CRASH vt PL $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py LS_ctrl_VB PQ2_VB PQ2_ns PQ2_nj >> ../../../$LOG 2>&1
echo "S42L_DONE $(date -Is)" >> ../../../$LOG
