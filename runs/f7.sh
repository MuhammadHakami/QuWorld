#!/bin/bash
# F7 (registered 2026-10-12 before the §42t full-VT result): iff F6 was built and uploaded (xh passed) and F5c finished -> F7 = F5c + the
# xh step (F6's arm, or the step-4 winner if it passes its rule) with XPA s1-3 -> agreement vs F5c -> upload (deletes 978859 = F3).
# Expected F7 public = F5c public + (F6L public - F4 public) (+ the step-4 delta if a new arm wins), stated in doc 35 once known.
# §42u (2026-10-13): + li2 (second li pass), which gained +0.000067 on public (F6L).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f7.log
until grep -q "^F6_DONE\|^F6_SKIP\|^F6_FAIL\|^CRASH" $L/f6_xh.log 2>/dev/null; do sleep 60; done
grep -q "^F6_DONE" $L/f6_xh.log || { echo "F7_SKIP (no F6) $(date -Is)" >> $LOG; exit 0; }
until grep -q "^F5C_DONE\|^F5C_FAIL\|^CRASH" $L/f5c.log 2>/dev/null; do sleep 60; done
grep -q "^F5C_DONE" $L/f5c.log || { echo "F7_SKIP (no F5c) $(date -Is)" >> $LOG; exit 0; }
ARM=$(grep -o "^\[f6\] arm [A-Za-z0-9]*" $L/f6_xh.log | tail -1 | awk '{print $3}')
until grep -q "^S42T_VT3_DONE\|^CRASH" $L/s42t_vt3.log 2>/dev/null; do sleep 60; done     # §42t step 4: a better xh variant?
NEW=$(python3 - $L/s42t_vt3.log <<'PY'
import re, sys
t = open(sys.argv[1]).read(); pat = r"\[vtpp\] PQmix~g_(\w+) vs PQmix~g_(\w+): S \S+ vs \S+ delta ([-+0-9.]+) \[([-+0-9.]+), ([-+0-9.]+)\] \| r2_psi \S+ r2_qb \S+ dlcfs ([-+0-9.]+)"
vs_c = {m[0]: (float(m[2]), float(m[3]), float(m[5])) for m in re.findall(pat, t) if m[1] == "VBc"}
vs_e = {m[0]: float(m[3]) for m in re.findall(pat, t) if m[1] == "VBxhe10e"}
ok = [(v[0], a) for a, v in vs_c.items() if a in ("VBxhe0e", "VBxhe0w75e") and v[1] > 0 and v[2] <= 0 and vs_e.get(a, -1) > 0]
print(max(ok)[1] if ok else "keep")
PY
)
[ "$NEW" != keep ] && ARM=$NEW
echo "[f7] step-4 decision: $NEW -> arm $ARM $(date -Is)" >> $LOG
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2S_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
KW=$(python3 -c "import json; k = json.load(open('$L/xh_arms.json'))['$ARM']; k['_steps'] = k['_steps'] + ['li2']; print(json.dumps(k))")   # §42u: + second li pass (F6L)
echo "[f7] arm $ARM kw $KW $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f7 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F7 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "F7_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F7/diii_d_public_test.npz runs/sophelio_sota/submission_F5c/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F7_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
set -a; . ./.env; set +a
curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/978859/" -o /dev/null -w "[f7] delete 978859 http %{http_code}\n" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F7/submission.zip F7 "F7: F5c + continuous X-point placement (dedicated X-point heads) + second li pass" >> $LOG 2>&1
echo "F7_DONE $(date -Is)" >> $LOG
