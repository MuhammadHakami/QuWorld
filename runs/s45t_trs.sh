#!/bin/bash
# §45t (Fusion): continuous Rgeo compensation of the column-quantised tri ("trs" step) - paired arms ctrl / trs / trsli in ONE VT inference
# (deployed F16K chain steps, TRAIN-only heads as S44H). VT_SHOTS: the 150 §44g / §44h shots by default; SUF tags the run.
#   setsid nohup runs/s45t_trs.sh [SUF] [shot list] &
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; LOG=runs/strat/fusion/s45t.log
SUF=${1:-150}; SH150=$(grep -o -- '--vt-shots "[0-9,]*"' runs/strat/fusion/s44h_jgn_end.sh | cut -d'"' -f2); SHOTS=${2:-$SH150}
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
KWS="{\"ctrl$SUF\": {\"_steps\": [$C], $X}, \"trs$SUF\": {\"_steps\": [$C, \"trs\"], $X}, \"trsli$SUF\": {\"_steps\": [$C, \"trs\", \"trsli\"], $X}, \"trsc10$SUF\": {\"_steps\": [$C, \"trs\"], $X, \"trs\": {\"cap\": 0.010}}, \"trsc05$SUF\": {\"_steps\": [$C, \"trs\"], $X, \"trs\": {\"cap\": 0.005}}}"
echo "[s45t] start $SUF $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45t 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S45T$SUF --psi psiF --psis "PQmix~g" \
  --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
  --heads "head_boundary_v8_J1RL_s1*3+head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2+head_boundary_v8_J1QH_s1+head_boundary_v8_J1QH_s2+head_boundary_v8_J1QH_s3" \
  --xpheads head_xp_XP1_s1+head_xp_XP1_s2+head_xp_XP1_s3 --kws "$KWS" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 || { echo "CRASH s45t vt $SUF $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_ctrl$SUF" "PQmix~g_trs$SUF" "PQmix~g_trsli$SUF" "PQmix~g_trsc10$SUF" "PQmix~g_trsc05$SUF" >> $LOG 2>&1
echo "S45T_DONE $SUF $(date -Is)" >> $LOG
