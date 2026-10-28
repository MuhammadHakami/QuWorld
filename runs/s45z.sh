#!/bin/bash
# §45z (doc 35) diagnostic ceiling: the f17eq VT arm (§45y) with the TRUE PCA coefficients [64, 256) added to the trunk psi before the chain.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; LOG=runs/strat/fusion/s45z.log
SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' runs/strat/fusion/s44h_jgn_end.sh | cut -d'"' -f2)
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R="head_boundary_v8_J1RLCS_s1*24"; H="route:$N+$QPI+$QHCS|$R+$QPI+$QHCS"
echo "[s45z] start $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_HIORACLE=64 QW_GPU_GB=30 runs/capped.sh s45z 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S45Zhi64 --psi psiF \
  --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
  --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"hi64\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
  || { echo "CRASH s45z $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17eq" "PQmix~g_hi64" >> $LOG 2>&1
echo "S45Z_DONE $(date -Is)" >> $LOG
