#!/bin/bash
# §48o stage A / B (doc 35, registered before any S result): psi temporal share - per-phase (PQmixV32p) and global 0.95 (PQmixV32w95) vs f20Pc on A150
# (CI lower > 0), best passing arm confirmed on B150 (point > 0 vs PQmixV32~g in the same inference).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48o.log
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
H="route:$N+$QHCS+$VV|$R_+$QHCS+$VV"; QB="head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6"
vt() { flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48o 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag $1 --psi psiF \
  --psis "$2" --guard psiF:0.25 --qb qh50 --qbheads "$QB" --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"$3\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$4" >> $LOG 2>&1; }
A=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2); B=$(cat $L/vt_shots_B150.txt)
echo "[s48o] stage A start $(date -Is)" >> $LOG
vt S48OA "PQmixV32p~g,PQmixV32w95~g" f20Po "$A" || { echo "S48O_FAIL vt A $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmixV32~g_f20Pc" "PQmixV32p~g_f20Po" "PQmixV32w95~g_f20Po" >> $LOG 2>&1
best=""; bd=-1
for s in PQmixV32p PQmixV32w95; do
  ln=$(grep "$s~g_f20Po vs PQmixV32~g_f20Pc" $LOG | tail -1); lo=$(sed 's/.*\[\([-+0-9.]*\),.*/\1/' <<< "$ln"); dl=$(grep -o "delta [-+0-9.]*" <<< "$ln" | cut -d' ' -f2)
  echo "[s48o] A $s delta ${dl:-?} CI lower ${lo:-?}" >> $LOG
  awk -v l="$lo" -v d="$dl" -v b="$bd" 'BEGIN{exit !(l > 0 && d > b)}' && { best=$s; bd=$dl; }
done
[ -n "$best" ] || { echo "S48O_REJECT (no A150 candidate; F20K share 0.5 stays) $(date -Is)" >> $LOG; exit 0; }
vt S48OB "PQmixV32~g,$best~g" f20Bo "$B" || { echo "S48O_FAIL vt B $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmixV32~g_f20Bo" "$best~g_f20Bo" >> $LOG 2>&1
dB=$(grep "$best~g_f20Bo vs PQmixV32~g_f20Bo" $LOG | tail -1 | grep -o "delta [-+0-9.]*" | cut -d' ' -f2)
if awk -v d="$dB" 'BEGIN{exit !(d > 0)}'; then echo "S48O_ADOPT $best (A150 delta $bd, B150 delta $dB) $(date -Is)" >> $LOG
else echo "S48O_REJECT ($best A150 delta $bd, B150 delta ${dB:-?}; F20K share 0.5 stays) $(date -Is)" >> $LOG; fi
