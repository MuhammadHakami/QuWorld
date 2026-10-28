#!/bin/bash
# §43h (registered before any §43h training result): register heads with J1RL's inputs, trained on the layered fast register path.
# After F12K (GPU free): two TRAIN-only register heads in parallel, both QW_QFAST=1, J1QPIpi architecture / lr / 60 epochs / seed 1:
#   J1QPIpiF_s1 = J1QPIpi exactly (magfix inputs)              -> fast-path training check vs the slow-path J1QPIpi_s1
#   J1QRL_s1    = + thz + hfull + liphys + qphys2 (J1RL inputs)  -> register head with the physics-ratio inputs
# Read-outs on all VT shots (vt_eval shape, paired shot bootstrap): singles vs J1QPIpi_s1; ensembles at F8's register share 1/7
# A = J1RL s1*3 + s2*3 + J1QPIpi_s1*1 vs B = same with J1QRL_s1 (C = J1QRL_s1*2, share 2/8: descriptive only).
# Rule: fastok iff |J1QPIpiF_s1 - J1QPIpi_s1| <= 0.0004; pass iff fastok and B - A > 0 with CI > 0 -> all-data J1QRLA s1-2
# (fast path) -> F13 = F12 with J1QRLA replacing J1QPIpiAH (alone) -> agreement vs F12 -> F13K (kw2). No upload here.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43h.log
until grep -q "^F12K_DONE\|^F12K_SKIP\|^CRASH" $L/f12k.log 2>/dev/null; do sleep 60; done
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
RL="STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2"
reg() {   # $1 tag  $2 seed  $3.. extra env
  local tag=$1 s=$2; shift 2
  [ -f $L/head_boundary_v8_${tag}_s$s.pt ] && return 0
  env STRAT_CUDA=0 QW_QFAST=1 QW_QCOMPILE=1 QW_MAGFIX=X_magfix QW_GPU_GB=20 "$@" runs/capped.sh q$tag$s 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py \
    --fixed "$PK" --tag $tag --epochs 60 --seed $s > $L/${tag}_s$s.log 2>&1 && echo "${tag}_s${s}_TRAINED $(date -Is)" >> $LOG || echo "CRASH $tag s$s $(date -Is)" >> $LOG
}
echo "[s43h] start $(date -Is)" >> $LOG
reg J1QPIpiF 1 & reg J1QRL 1 $RL & wait
grep -q "^CRASH" $LOG && exit 1
H=head_boundary_v8; E="$H"_J1RL_s1*3+"$H"_J1RL_s2*3
flock runs/strat/vt.lock .venv/bin/python paper/strat/fusion/vt_eval.py shape ${H}_J1QPIpi_s1 ${H}_J1QPIpiF_s1 ${H}_J1QRL_s1 >> $LOG 2>&1 || { echo "CRASH vt singles $(date -Is)" >> $LOG; exit 1; }
flock runs/strat/vt.lock .venv/bin/python paper/strat/fusion/vt_eval.py shape "$E+${H}_J1QPIpi_s1*1" "$E+${H}_J1QRL_s1*1" "$E+${H}_J1QRL_s1*2" >> $LOG 2>&1 \
  || { echo "CRASH vt ensembles $(date -Is)" >> $LOG; exit 1; }
read FASTOK PASS < <(python3 - $LOG <<'PY'
import re, sys
t = open(sys.argv[1]).read(); num = r"([-+0-9.]+)"
f = re.search(r"\[vt\] head_boundary_v8_J1QPIpiF_s1 vs head_boundary_v8_J1QPIpi_s1: " + num, t)
e = re.search(r"\[vt\] \S+J1QRL_s1\*1 vs \S+J1QPIpi_s1\*1: " + num + r" \[" + num + ", " + num + r"\]", t)
fastok = "yes" if f and abs(float(f.group(1))) <= 0.0004 else "no"
print(fastok, "yes" if fastok == "yes" and e and float(e.group(1)) > 0 and float(e.group(2)) > 0 else "no")
PY
)
echo "[s43h] fast path ok: $FASTOK; J1QRL ensemble pass: $PASS $(date -Is)" >> $LOG
[ "$PASS" = yes ] || { echo "S43H_DONE (no F13) $(date -Is)" >> $LOG; exit 0; }
reg J1QRLA 1 $RL STRAT_ALL=1 & reg J1QRLA 2 $RL STRAT_ALL=1 & wait
grep -q "^CRASH" $LOG && exit 1
H13=$(grep -o 'heads head_boundary[^ ]*' $L/s43g.log | head -1 | cut -d' ' -f2 | sed 's/J1QPIpiAH/J1QRLA/g')
grep -q "J1QRLA_s1\*2" <<< "$H13" || { echo "F13_FAIL heads parse $(date -Is)" >> $LOG; exit 1; }
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
python3 -c "import json, sys; json.loads(sys.argv[1])" "$KW" || { echo "F13_FAIL kw parse $(date -Is)" >> $LOG; exit 1; }
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
echo "[f13] heads $H13 $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f13 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F13 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H13" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F13_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F13/diii_d_public_test.npz runs/sophelio_sota/submission_F12/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F13_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F13_BUILT $(date -Is)" >> $LOG
QW_GPU_GB=12 runs/capped.sh kw2f13 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F13 --tag F13K --heads "$H13" >> $LOG 2>&1 \
  && echo "F13K_DONE $(date -Is)" >> $LOG || echo "CRASH F13K $(date -Is)" >> $LOG
echo "S43H_DONE $(date -Is)" >> $LOG
