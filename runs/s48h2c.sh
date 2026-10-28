#!/bin/bash
# §48h stage-2 decision arm f20Pc (amendment registered before any VT result): psi PQmixV32~g (PSIV60 s1-2 on components 0-31, MLP trunks alone above),
# heads F18K - J1QPIpi; adopt iff paired CI lower > 0 and D_LCFS not worse vs PQmix~g_f17V60. Runs after s48h2.sh (f20Pa / f20Pb diagnostics).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48h.log
until grep -qE "^S48H2_DONE|^S48H2_FAIL" $LOG; do sleep 120; done
grep -q "^S48H2_FAIL training" $LOG && { echo "S48H2C_FAIL (no PSIV60) $(date -Is)" >> $LOG; exit 1; }
SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2)
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
H="route:$N+$QHCS+$VV|$R_+$QHCS+$VV"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48hf20Pc 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S48Hf20Pc --psi psiF \
  --psis "PQmixV32~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
  --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"f20Pc\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
  || { echo "S48H2C_FAIL vt $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17V60" "PQmixV32~g_f20Pc" >> $LOG 2>&1
line=$(grep "PQmixV32~g_f20Pc vs PQmix~g_f17V60" $LOG | tail -1); lo=$(sed 's/.*\[\([-+0-9.]*\),.*/\1/' <<< "$line"); dl=$(grep -o "dlcfs [-+0-9.]*" <<< "$line" | cut -d' ' -f2)
if awk -v l="$lo" -v d="$dl" 'BEGIN{exit !(l > 0 && d <= 0)}'; then echo "S48H2C_ADOPT F20K (f20Pc CI lower $lo, dlcfs $dl) $(date -Is)" >> $LOG
else echo "S48H2C_REJECT (f20Pc CI lower ${lo:-?}, dlcfs ${dl:-?}; F18K stays) $(date -Is)" >> $LOG; fi
