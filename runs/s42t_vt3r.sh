#!/bin/bash
# §42t step 4 rerun (arms / heads / rule unchanged): the first VT run was killed by memguard at 09:59 (avail 26 GB with F5c's public build).
# Runs after F7's build so that no public build and VT run overlap. Log: s42t_vt3r.log (F7 already read the step-4 CRASH as "keep VBxhe10").
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42t_vt3r.log
until grep -q "^F7_DONE\|^F7_FAIL\|^F7_SKIP" $L/f7.log 2>/dev/null; do sleep 60; done
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS=$(python3 -c "import json; A=json.load(open('$L/xh_arms.json')); print(json.dumps({k: A[k] for k in ('VBxhe10e', 'VBxhe0e', 'VBxhe0w75e')}))")
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42tvt3r 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XH3 \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3 \
  --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH vt XH3r $(date -Is)" >> $LOG; exit 1; }
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBc" "PQmix~g_VBxhe10" "PQmix~g_VBxhe10e" "PQmix~g_VBxhe0e" "PQmix~g_VBxhe0w75e" >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBxhe10e" "PQmix~g_VBxhe0e" "PQmix~g_VBxhe0w75e" >> ../../../$LOG 2>&1
echo "S42T_VT3R_DONE $(date -Is)" >> ../../../$LOG
