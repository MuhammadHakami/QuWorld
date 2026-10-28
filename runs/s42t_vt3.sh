#!/bin/bash
# §42t step 4 (registered before any result): XP1 s2/s3 (TRAIN only) -> after F6's build: VT arms VBxhe10e / VBxhe0e / VBxhe0w75e with the
# XP1 s1-3 ensemble; compared with the saved step-3 PQmix~g_VBc / PQmix~g_VBxhe10 parts. Rule in doc 35.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42t_vt3.log
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
for s in 2 3; do
  [ -f $L/head_xp_XP1_s$s.pt ] || env $J STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_XP1 QW_GPU_GB=30 \
    runs/capped.sh hXP1 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s xp >> $LOG 2>&1 || { echo "CRASH XP1 s$s $(date -Is)" >> $LOG; exit 1; }
  echo "XP1_s${s}_TRAINED $(date -Is)" >> $LOG
done
until grep -q "^F6_DONE\|^F6_FAIL\|^F6_SKIP\|^CRASH" $L/f6_xh.log 2>/dev/null; do sleep 60; done
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS=$(python3 -c "import json; A=json.load(open('$L/xh_arms.json')); print(json.dumps({k: A[k] for k in ('VBxhe10e', 'VBxhe0e', 'VBxhe0w75e')}))")
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42tvt3 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XH3 \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3 \
  --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH vt XH3 $(date -Is)" >> $LOG; exit 1; }
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBc" "PQmix~g_VBxhe10" "PQmix~g_VBxhe10e" "PQmix~g_VBxhe0e" "PQmix~g_VBxhe0w75e" >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBxhe10e" "PQmix~g_VBxhe0e" "PQmix~g_VBxhe0w75e" >> ../../../$LOG 2>&1
echo "S42T_VT3_DONE $(date -Is)" >> ../../../$LOG
