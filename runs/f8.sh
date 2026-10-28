#!/bin/bash
# F8 (registered 2026-10-13 before the F7 / F7C / step-4 / §42x results) = the FINAL-CANDIDATE configuration on public (repro/fusion):
# F7 with the QHAR2 q/b heads (rebuild budget; QHAR2S's plasma-shape inputs need 19 extra source heads) + the in-pipeline q/b clamp (§42z)
# + the step-4 xh arm if it passed (F7's rule) + kw2 iff runs/strat/fusion/kw2_adopt.txt says "yes" (decided from F7KC vs F7C on public).
# Builds ALONE: after the step-4 VT rerun and §42x's read-out. Agreement vs F7C. No upload here (UTC-day budget; decided by hand).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f8.log
until grep -q "^S42T_VT3R_DONE\|^CRASH" $L/s42t_vt3r.log 2>/dev/null; do sleep 60; done
until grep -q "^S42X_DONE\|^CRASH" $L/s42x.log 2>/dev/null; do sleep 60; done
NEW=$(python3 - $L/s42t_vt3r.log <<'PY'
import re, sys
t = open(sys.argv[1]).read(); pat = r"\[vtpp\] PQmix~g_(\w+) vs PQmix~g_(\w+): S \S+ vs \S+ delta ([-+0-9.]+) \[([-+0-9.]+), ([-+0-9.]+)\] \| r2_psi \S+ r2_qb \S+ dlcfs ([-+0-9.]+)"
vs_c = {m[0]: (float(m[2]), float(m[3]), float(m[5])) for m in re.findall(pat, t) if m[1] == "VBc"}
vs_e = {m[0]: float(m[3]) for m in re.findall(pat, t) if m[1] == "VBxhe10e"}
ok = [(v[0], a) for a, v in vs_c.items() if a in ("VBxhe0e", "VBxhe0w75e") and v[1] > 0 and v[2] <= 0 and vs_e.get(a, -1) > 0]
print(max(ok)[1] if ok else "keep")
PY
)
ARM=VBxhe10; [ "$NEW" != keep ] && ARM=$NEW            # the step-4 arms (VBxhe0e / VBxhe0w75e) are plain kw configs
KW2=$(cat $L/kw2_adopt.txt 2>/dev/null || echo no)
KW=$(python3 -c "
import json; A = json.load(open('$L/xh_arms.json')); k = A['$ARM']; k['_steps'] = k['_steps'] + ['li2'] + (['kw2'] if '$KW2' == 'yes' else []); print(json.dumps(k))")
echo "[f8] step-4: $NEW -> arm $ARM; kw2 $KW2; kw $KW $(date -Is)" >> $LOG
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f8 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F8 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F8_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F8/diii_d_public_test.npz runs/sophelio_sota/submission_F7C/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F8_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F8_BUILT $(date -Is)" >> $LOG
