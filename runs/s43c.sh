#!/bin/bash
# §43c (registered before any result): VT check of the exact X-point labels. XP1E s2-3 (TRAIN only, after F11's build) -> one VT run with the
# deployed xh arm (gate 0, w 0.75) on the XP1E s1-3 ensemble (arm VBxhe0w75x) -> paired vs the saved PQmix~g_VBxhe0w75e (XP1 s1-3, PCA labels).
# Read-out only (F11 = F8 + XPEA is already the public probe); expected delta 0..+0.00004, D_LCFS -1..-3%.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43c.log
until grep -q "^F11_BUILT\|^F11_FAIL\|^CRASH" $L/s43b.log 2>/dev/null; do sleep 60; done
J="STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix QW_GPU_GB=16 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1"
for s in 2 3; do
  [ -f $L/head_xp_XP1E_s$s.pt ] || { env $J STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 STRAT_OUT_TAG=_XP1E runs/capped.sh hXP1E 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s xp >> $L/s43c_XP1E.log 2>&1 && echo "XP1E_s${s}_TRAINED $(date -Is)" >> $LOG || echo "CRASH XP1E s$s $(date -Is)" >> $LOG; } &
done
wait
grep -q "^CRASH" $LOG && exit 1
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS=$(python3 -c "import json; A=json.load(open('$L/xh_arms.json')); print(json.dumps({'VBxhe0w75x': A['VBxhe0w75x']}))")
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s43cvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XH4 \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1E_s1+head_xp_XP1E_s2+head_xp_XP1E_s3 \
  --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH vt XH4 $(date -Is)" >> $LOG; exit 1; }
(cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBxhe0w75e" "PQmix~g_VBxhe0w75x") >> $LOG 2>&1
echo "S43C_DONE $(date -Is)" >> $LOG
