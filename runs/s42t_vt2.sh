#!/bin/bash
# §42t step 3 (registered after the 2-shot smoke, before any full-VT result): the smoke showed tri_top / tri_bot +0.005 but kappa -0.012..-0.018
# from the late RZ move (kw had already fit kappa). Arms (runs/strat/fusion/xh_arms.json): VBc, VBxh10 (registered), VBxhR10 / VBxhR0 (R only),
# VBxhe10 (RZ before kw / xp / jgn). Rule: adopt the best arm by delta iff its S delta vs VBc > 0 with CI > 0 and D_LCFS not worse.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42t_vt2.log
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS=$(python3 -c "import json; print(json.dumps(json.load(open('$L/xh_arms.json'))))")
env QW_GPU_GB=20 runs/capped.sh s42tsm 30 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --vt-n 2 --tag XHs \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1 --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH smoke $(date -Is)" >> $LOG; exit 1; }
(cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBc" "PQmix~g_VBxh10" "PQmix~g_VBxhR10" "PQmix~g_VBxhR0" "PQmix~g_VBxhe10" | sed 's/^/smoke /') >> $LOG 2>&1
echo "XH2_SMOKE_OK $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42tvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XH \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1 --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH vt XH $(date -Is)" >> $LOG; exit 1; }
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBc" "PQmix~g_VBxh10" "PQmix~g_VBxhR10" "PQmix~g_VBxhR0" "PQmix~g_VBxhe10" >> ../../../$LOG 2>&1
echo "S42T_VT2_DONE $(date -Is)" >> ../../../$LOG
