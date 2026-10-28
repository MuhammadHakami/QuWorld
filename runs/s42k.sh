#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42k.log
until grep -q "^S42H_DONE" $L/s42h.log 2>/dev/null; do sleep 60; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
VB='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42kvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag KS \
  --psi psiF --psis "psiF,psiF~s" --shrink $L/coef_reliability.npz --qb qh50 --qbheads "$Q" --heads "$J" --kws "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" \
  --axis head >> $LOG 2>&1 || echo "CRASH vt KS $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py psiF_raw "psiF~s_raw" >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py psiF_VB "psiF~s_VB" >> ../../../$LOG 2>&1
echo "S42K_DONE $(date -Is)" >> ../../../$LOG
