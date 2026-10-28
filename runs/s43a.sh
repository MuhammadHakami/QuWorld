#!/bin/bash
# §43a (registered before any exact-label result): exact TRAIN labels. Screen J1RLE_s1 (J1RL recipe on v9 labels, TRAIN only) vs J1RL_s1 on
# all VT shots + XP1E_s1 (X-point heads on exact X-point labels) vs XP1_s1 (X-point error, 60 oracle shots). Rule: iff mean-7 delta >=
# +0.0002 with CI > 0 -> all-data J1RLEA s1-8 (+ XPEA s1-3 iff XP1E's X-point error is not worse) -> F11 = F8 with them (alone) -> agreement
# vs F8. Two head trainings at a time (each QW_GPU_GB=16); the public build alone. No upload here.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43a.log
until grep -q "^LABELS_EXACT_DONE" $L/labels_exact.log 2>/dev/null && grep -q "^XPL_EXACT_DONE" $L/xpl_exact.log 2>/dev/null; do
  grep -q "Traceback" $L/labels_exact.log $L/xpl_exact.log 2>/dev/null && { echo "CRASH labels $(date -Is)" >> $LOG; exit 1; }; sleep 60; done
J="STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix QW_GPU_GB=16"
R="STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1"
trainhead() {   # $1 tag  $2 seed  $3 mode  $4.. extra env
  local tag=$1 s=$2 mode=$3; shift 3
  env $J $R "$@" STRAT_OUT_TAG=_$tag runs/capped.sh h$tag 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s $mode >> $L/s43a_$tag.log 2>&1 \
    && echo "${tag}_s${s}_TRAINED $(date -Is)" >> $LOG || echo "CRASH $tag s$s $(date -Is)" >> $LOG
}
[ -f $L/head_boundary_v9_J1RLE_s1.pt ] || trainhead J1RLE 1 shape STRAT_LABEL_SUFFIX=_v9 STRAT_SCAL_W=9.142857 &
[ -f $L/head_xp_XP1E_s1.pt ] || trainhead XP1E 1 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 &
wait
grep -q "^CRASH" $LOG && exit 1
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v9_J1RLE_s1 >> $LOG 2>&1 || { echo "CRASH vt shape $(date -Is)" >> $LOG; exit 1; }
(cd paper/strat/fusion && ../../../.venv/bin/python xp_blend.py head_xp_XP1_s1 && ../../../.venv/bin/python xp_blend.py head_xp_XP1E_s1) >> $LOG 2>&1 || echo "CRASH xp_blend $(date -Is)" >> $LOG
read PASS XPOK < <(python3 - $LOG <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"\[vt\] head_boundary_v9_J1RLE_s1 vs head_boundary_v8_J1RL_s1: ([-+0-9.]+) \[([-+0-9.]+), ([-+0-9.]+)\]", t)
p = "yes" if m and float(m.group(1)) >= 0.0002 and float(m.group(2)) > 0 else "no"
w = re.findall(r"\[xpb\] w 1.00: median ([0-9.]+) mm, p75 [0-9.]+, p90 [0-9.]+, rms ([0-9.]+)", t)
x = "yes" if len(w) == 2 and float(w[1][0]) <= float(w[0][0]) and float(w[1][1]) <= float(w[0][1]) else "no"
print(p, x)
PY
)
echo "[s43a] shape pass: $PASS; exact X-point labels not worse: $XPOK $(date -Is)" >> $LOG
[ "$PASS" = yes ] || { echo "S43A_DONE (no F11) $(date -Is)" >> $LOG; exit 0; }
A="STRAT_ALL=1"
for pair in "1 2" "3 4" "5 6" "7 8"; do
  set -- $pair
  [ -f $L/head_boundary_v9_J1RLEA_s$1.pt ] || trainhead J1RLEA $1 shape STRAT_LABEL_SUFFIX=_v9 STRAT_SCAL_W=9.142857 $A &
  [ -f $L/head_boundary_v9_J1RLEA_s$2.pt ] || trainhead J1RLEA $2 shape STRAT_LABEL_SUFFIX=_v9 STRAT_SCAL_W=9.142857 $A &
  wait
done
XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
if [ "$XPOK" = yes ]; then
  [ -f $L/head_xp_XPEA_s1.pt ] || trainhead XPEA 1 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 $A &
  [ -f $L/head_xp_XPEA_s2.pt ] || trainhead XPEA 2 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 $A &
  wait
  [ -f $L/head_xp_XPEA_s3.pt ] || trainhead XPEA 3 xp STRAT_LABEL_SUFFIX=_v8 STRAT_XP_LABELS=xp_labels_v9 $A
  XH="head_xp_XPEA_s1+head_xp_XPEA_s2+head_xp_XPEA_s3"
fi
grep -q "^CRASH" $LOG && exit 1
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
python3 -c "import json, sys; json.loads(sys.argv[1])" "$KW" || { echo "F11_FAIL kw parse $(date -Is)" >> $LOG; exit 1; }
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v9_J1RLEA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//')
echo "[f11] heads J1RLEA x8, xh $XH, kw $KW $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f11 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F11 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F11_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F11/diii_d_public_test.npz runs/sophelio_sota/submission_F8/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F11_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F11_BUILT $(date -Is)" >> $LOG
echo "S43A_DONE $(date -Is)" >> $LOG
