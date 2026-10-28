#!/bin/bash
# §45y (doc 35): chain-level VT confirmation of the adopted F17K vs F16K - two VT inferences on the 150 §45t shots (TRAIN-only heads, the
# deployed F16K chain), f16eq (register J1QH, X-point XP1_s1) vs f17eq (J1QHCS, XP1CS_s1), paired by vt_pp_compare.py.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; LOG=runs/strat/fusion/s45y.log
SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' runs/strat/fusion/s44h_jgn_end.sh | cut -d'"' -f2)
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QH="head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
QHCS=${QH//J1QH_/J1QHCS_}; N="head_boundary_v8_J1RL_s1*24"; R="head_boundary_v8_J1RLCS_s1*24"
echo "[s45y] start $(date -Is)" >> $LOG
for arm in f16eq f17eq; do
  if [ $arm = f16eq ]; then H="route:$N+$QPI+$QH|$R+$QPI+$QH"; XP=head_xp_XP1_s1; else H="route:$N+$QPI+$QHCS|$R+$QPI+$QHCS"; XP=head_xp_XP1CS_s1; fi
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45y 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S45Y$arm --psi psiF \
    --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
    --heads "$H" --xpheads $XP --kws "{\"$arm\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
    || { echo "CRASH s45y $arm $(date -Is)" >> $LOG; exit 1; }
done
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f16eq" "PQmix~g_f17eq" >> $LOG 2>&1
echo "S45Y_DONE $(date -Is)" >> $LOG
