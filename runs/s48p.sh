#!/bin/bash
# §48p (doc 35, registered after §48o stage A, before any B150 read of the share): one confirmatory read of temporal psi share 0.95 on the
# disjoint B150 shots. After §48m's verdict: REJECT/FAIL -> PQmixV32w95 vs PQmixV32 (F20K); ADOPT -> PQmixV32x3w95 vs PQmixV32x3 (F21K).
# Adopt share 0.95 iff the B150 paired 95% CI lower > 0.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48p.log
echo "[s48p] start $(date -Is)" >> $LOG
until grep -qE "^S48M_(ADOPT|REJECT|FAIL)" $L/s48m.log; do sleep 120; done
if grep -q "^S48M_ADOPT" $L/s48m.log; then BASE=PQmixV32x3; ARM=PQmixV32x3w95; else BASE=PQmixV32; ARM=PQmixV32w95; fi
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
H="route:$N+$QHCS+$VV|$R_+$QHCS+$VV"; QB="head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6"
B=$(cat $L/vt_shots_B150.txt)
echo "[s48p] base $BASE arm $ARM on B150 $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48p 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S48PB --psi psiF \
  --psis "$BASE~g,$ARM~g" --guard psiF:0.25 --qb qh50 --qbheads "$QB" --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"f20Bp\": {\"_steps\": [$C], $X}}" \
  --axis head --qbclamp --vt-shots "$B" >> $LOG 2>&1 || { echo "S48P_FAIL vt $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "$BASE~g_f20Bp" "$ARM~g_f20Bp" >> $LOG 2>&1
ln=$(grep "$ARM~g_f20Bp vs $BASE~g_f20Bp" $LOG | tail -1); lo=$(sed 's/.*\[\([-+0-9.]*\),.*/\1/' <<< "$ln"); dl=$(grep -o "delta [-+0-9.]*" <<< "$ln" | cut -d' ' -f2)
if awk -v l="$lo" 'BEGIN{exit !(l > 0)}'; then echo "S48P_ADOPT $ARM (B150 delta $dl, CI lower $lo) $(date -Is)" >> $LOG
else echo "S48P_REJECT $ARM (B150 delta ${dl:-?}, CI lower ${lo:-?}; share 0.5 stays) $(date -Is)" >> $LOG; fi
