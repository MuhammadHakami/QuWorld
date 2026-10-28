#!/bin/bash
# §48i' (doc 35: the reduced re-run of §48i, registered after stage A was stopped unread): post-processing chain + temporal psi share re-tune for F20K - stage A (standard 150 VT shots, 20 arms),
# stage B (disjoint 150 shots vt_shots_B150.txt) confirms; adopt iff paired CI lower > 0 on both.
cd /home/mohhakami/Documents/paper/quworld; until grep -qE "^S48H3_(READY|FAIL)|^\[s48h3\] stage 2 did not adopt" runs/strat/fusion/s48h.log; do sleep 300; done; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48i2.log
echo "[s48i] start $(date -Is)" >> $LOG
X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
STEPS=(shift bw xhe kw xp jgn li li2 kw2)
kws() {                                                                       # kws <suffix> <arm list> -> JSON of named kw arms
  local sfx=$1; shift; local out="" a st
  for a in "$@"; do
    st=""; for s in "${STEPS[@]}"; do [ "$a" = "n$s" ] && continue; st="$st\"$s\", "; done; st="${st%, }"
    out="$out\"$a$sfx\": {\"_steps\": [$st], $X}, "
  done; echo "{${out%, }}"; }
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
H="route:$N+$QHCS+$VV|$R_+$QHCS+$VV"; QB="head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6"
vt() {                                                                        # vt <tag> <psis> <shots> <kws json>
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48i 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag $1 --psi psiF \
    --psis "$2" --guard psiF:0.25 --qb qh50 --qbheads "$QB" --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "$4" --axis head --qbclamp --vt-shots "$3" >> $LOG 2>&1; }
cmp_() { .venv/bin/python paper/strat/fusion/vt_pp_compare.py "$1" "$2" >> $LOG 2>&1; grep "$2 vs $1" $LOG | tail -1 | sed 's/.*\[\([-+0-9.]*\),.*/\1/'; }
A=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2); B=$(cat $L/vt_shots_B150.txt)
ARMS=(p0 nkw njgn nshift)
vt S48IA2 "PQmixV32~g,PQmixV32h~g" "$A" "$(kws "" "${ARMS[@]}")" || { echo "S48I_FAIL stage A $(date -Is)" >> $LOG; exit 1; }
CAND=()
for ps in PQmixV32 PQmixV32h; do for a in "${ARMS[@]}"; do
  [ "$ps$a" = PQmixV32p0 ] && continue
  lo=$(cmp_ "PQmixV32~g_p0" "$ps~g_$a"); echo "S48I_A $ps~g_$a CI lower ${lo:-?}" >> $LOG
  awk -v l="$lo" 'BEGIN{exit !(l > 0)}' && CAND+=("$ps:$a")
done; done
echo "[s48i] stage A candidates: ${CAND[*]:-none} $(date -Is)" >> $LOG
[ ${#CAND[@]} -gt 0 ] || { echo "S48I_DONE none (chain + share stay) $(date -Is)" >> $LOG; exit 0; }
# stage B: base + candidates (+ combination: the psi share of any candidate on PQmixV32h, all removed steps together)
SETS="PQmixV32~g"; ARMSB=(p0); for c in "${CAND[@]}"; do [ "${c%%:*}" = PQmixV32h ] && SETS="PQmixV32~g,PQmixV32h~g"; ARMSB+=("${c#*:}"); done
COMB=""; if [ ${#CAND[@]} -ge 2 ]; then
  COMB=$(printf "%s\n" "${CAND[@]}" | cut -d: -f2 | grep -v '^p0$' | sort -u | tr '\n' ' '); fi
KB=$(kws B $(printf "%s\n" "${ARMSB[@]}" | sort -u))
if [ -n "$COMB" ]; then                                                      # combination arm "cB": the chain without every removed step
  st=""; for s in "${STEPS[@]}"; do grep -qw "n$s" <<< "$COMB" && continue; st="$st\"$s\", "; done
  KB="${KB%\}}, \"cB\": {\"_steps\": [${st%, }], $X}}"; fi
vt S48IB2 "$SETS" "$B" "$KB" || { echo "S48I_FAIL stage B $(date -Is)" >> $LOG; exit 1; }
WIN=""; PSC=PQmixV32; [ "$SETS" != "PQmixV32~g" ] && PSC=PQmixV32h      # only stage-A candidates (+ the combination) are read in stage B
for c in "${CAND[@]}" ${COMB:+$PSC:c}; do ps=${c%%:*}; a=${c#*:}
  lo=$(cmp_ "PQmixV32~g_p0B" "$ps~g_${a}B"); echo "S48I_B $ps~g_${a}B CI lower ${lo:-?}" >> $LOG
  awk -v l="$lo" 'BEGIN{exit !(l > 0)}' && WIN="$WIN $ps:$a"
done
echo "S48I_DONE confirmed:${WIN:- none} $(date -Is)" >> $LOG
