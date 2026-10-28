#!/bin/bash
# F6 (registered with §42t; re-pointed 2026-10-12 22:15 before the full-VT result): iff an xh arm passes (S delta vs VBc > 0 with CI > 0,
# D_LCFS not worse; the best of the step-3 arms by delta, kw from xh_arms.json) -> XPA s1-3 (X-point heads, J1RL recipe, all data; F5c's
# training is paused meanwhile) -> F6 = F4 + that xh step with XPA s1-3 (isolates the X-point effect; F5c's heads join later) ->
# agreement vs F4 -> upload (deletes 976610 = F2). Expected F6 public = F4 0.995575 + the VT S delta (carry ~1), logged before the upload.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f6_xh.log
until grep -q "^S42T_VT2_DONE\|^CRASH" $L/s42t_vt2.log 2>/dev/null; do sleep 60; done
pick=$(python3 - $L/s42t_vt2.log <<'PY'
import re, sys
best = None
for m in re.finditer(r"\[vtpp\] PQmix~g_(VBxh\w+) vs PQmix~g_VBc: S \S+ vs \S+ delta ([-+0-9.]+) \[([-+0-9.]+), ([-+0-9.]+)\] \| r2_psi \S+ r2_qb \S+ dlcfs ([-+0-9.]+)", "".join(l for l in open(sys.argv[1]) if not l.startswith("smoke"))):
    arm, de, lo, dl = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(5))
    if lo > 0 and dl <= 0 and (best is None or de > best[1]):
        best = (arm, de)
print(f"{best[0]} {best[1]:+.6f}" if best else "none")
PY
)
echo "[f6] rule: $pick $(date -Is)" >> $LOG
[ "$pick" != none ] || { echo "F6_SKIP $(date -Is)" >> $LOG; exit 0; }
ARM=${pick%% *}; DELTA=${pick##* }
PAUSED=$(pgrep -f '[h]ead_boundary.py' | tr '\n' ' ')                      # F5c's running head: pause it while XPA trains
trap '[ -n "$PAUSED" ] && kill -CONT $PAUSED 2>/dev/null' EXIT
[ -n "$PAUSED" ] && kill -STOP $PAUSED && echo "[f6] paused F5c pids $PAUSED $(date -Is)" >> $LOG
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 1 2 3; do
  [ -f $L/head_xp_XPA_s$s.pt ] || env $E STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_XPA QW_GPU_GB=30 runs/capped.sh hXPA 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s xp >> $LOG 2>&1 || { echo "CRASH XPA s$s $(date -Is)" >> $LOG; exit 1; }
  echo "XPA_s${s}_TRAINED $(date -Is)" >> $LOG
done
[ -n "$PAUSED" ] && kill -CONT $PAUSED && echo "[f6] resumed F5c $(date -Is)" >> $LOG; PAUSED=""
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1AH_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
KW=$(python3 -c "import json; print(json.dumps(json.load(open('$L/xh_arms.json'))['$ARM']))")
echo "[f6] arm $ARM kw $KW" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f6 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F6 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "F6_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F6/diii_d_public_test.npz runs/sophelio_sota/submission_F4/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F6_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "[f6] expected public ~$(python3 -c "print(f'{0.995575 + float(\"$DELTA\"):.6f}')") (F4 0.995575 + VT delta $DELTA) $(date -Is)" >> $LOG
set -a; . ./.env; set +a
curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/976610/" -o /dev/null -w "[f6] delete 976610 http %{http_code}\n" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F6/submission.zip F6 "F6: F4 + continuous X-point placement (Newton saddle moved to dedicated X-point heads before the shape fits)" >> $LOG 2>&1
echo "F6_DONE $(date -Is)" >> $LOG
