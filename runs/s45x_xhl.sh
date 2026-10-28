#!/bin/bash
# §45x (Fusion): late xh (after jgn, before li / kw2) - arms xhl75 / xhl50 in ONE VT inference, paired vs the saved §45t ctrl$SUF parts.
#   setsid nohup runs/s45x_xhl.sh [SUF] [shot list] &      (SUF 150 = the §45t shots; a 593-shot confirmation needs its own ctrl arm)
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; LOG=runs/strat/fusion/s45x.log
SUF=${1:-150}; SH150=$(grep -o -- '--vt-shots "[0-9,]*"' runs/strat/fusion/s44h_jgn_end.sh | cut -d'"' -f2); SHOTS=${2:-$SH150}
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "xh", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
KWS="{\"xhl75$SUF\": {\"_steps\": [$C], $X, \"xh\": {\"gate\": 0.0, \"w\": 0.75}}, \"xhl50$SUF\": {\"_steps\": [$C], $X, \"xh\": {\"gate\": 0.0, \"w\": 0.5}}}"
[ "$SUF" = 150 ] || KWS="${KWS%\}}, \"ctrl$SUF\": {\"_steps\": [${C/\"xh\", /}], $X}}"
echo "[s45x] start $SUF $(date -Is) kws $KWS" >> $LOG
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45x 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S45X$SUF --psi psiF --psis "PQmix~g" \
  --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
  --heads "head_boundary_v8_J1RL_s1*3+head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2+head_boundary_v8_J1QH_s1+head_boundary_v8_J1QH_s2+head_boundary_v8_J1QH_s3" \
  --xpheads head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3 --kws "$KWS" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 || { echo "CRASH s45x vt $SUF $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_ctrl$SUF" "PQmix~g_xhl75$SUF" "PQmix~g_xhl50$SUF" >> $LOG 2>&1
echo "S45X_DONE $SUF $(date -Is)" >> $LOG
