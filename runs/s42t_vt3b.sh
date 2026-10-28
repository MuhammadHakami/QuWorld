#!/bin/bash
# §42t step 4 continuation (replaces s42t_vt3.sh's tail; arms, heads and rule unchanged): the VT run no longer takes runs/strat/vt.lock,
# so F5c's build is not blocked ~5 h behind it (two GPU jobs at most: this VT + F5c). Waits for the in-flight XP1 s2 training (PID $1).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42t_vt3.log
while kill -0 $1 2>/dev/null; do sleep 30; done
[ -f $L/head_xp_XP1_s2.pt ] || { echo "CRASH XP1 s2 (no checkpoint) $(date -Is)" >> $LOG; exit 1; }
echo "XP1_s2_TRAINED $(date -Is)" >> $LOG
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_xp_XP1_s3.pt ] || env $J STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_XP1 QW_GPU_GB=30 \
  runs/capped.sh hXP1 40 .venv/bin/python paper/strat/fusion/head_boundary.py 3 xp >> $LOG 2>&1 || { echo "CRASH XP1 s3 $(date -Is)" >> $LOG; exit 1; }
echo "XP1_s3_TRAINED $(date -Is)" >> $LOG
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS=$(python3 -c "import json; A=json.load(open('$L/xh_arms.json')); print(json.dumps({k: A[k] for k in ('VBxhe10e', 'VBxhe0e', 'VBxhe0w75e')}))")
env QW_GPU_GB=30 runs/capped.sh s42tvt3 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XH3 \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3 \
  --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH vt XH3 $(date -Is)" >> $LOG; exit 1; }
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBc" "PQmix~g_VBxhe10" "PQmix~g_VBxhe10e" "PQmix~g_VBxhe0e" "PQmix~g_VBxhe0w75e" >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBxhe10e" "PQmix~g_VBxhe0e" "PQmix~g_VBxhe0w75e" >> ../../../$LOG 2>&1
echo "S42T_VT3_DONE $(date -Is)" >> ../../../$LOG
