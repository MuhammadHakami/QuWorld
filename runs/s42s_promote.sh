#!/bin/bash
# §42s promotion + F5 (registered before the §42s result): iff QHR2S_s1 - QHR2_s1 >= +0.0002 with CI > 0 -> QHAR2S s1-3 (all data,
# qphys2 + qshape) -> F5 = F4 with QHAR2S s1-3 as the q/b heads -> agreement vs F4 -> upload (deletes 976250).
# Expected F5 public = F4 0.995575 + 0.15 x 1.5 x (VT q/b delta) (q/b weight x the F3 public carry), stated before the build.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42s_promote.log
until grep -q "^S42S_DONE\|^CRASH" $L/s42s.log 2>/dev/null; do sleep 60; done
ok=$(python3 - $L/s42s.log <<'PY'
import re, sys
m = re.search(r"QHR2S_s1 vs head_qb_QHR2_s1: ([-+0-9.]+) \[([-+0-9.]+), ", open(sys.argv[1]).read())
print(int(bool(m) and float(m.group(1)) >= 0.0002 and float(m.group(2)) > 0), m.group(1) if m else "")
PY
)
echo "[s42s] rule: $ok $(date -Is)" >> $LOG
[ "${ok%% *}" = 1 ] || { echo "S42S_PROMOTE_SKIP $(date -Is)" >> $LOG; exit 0; }
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
for s in 1 2 3; do
  [ -f $L/head_qb_QHAR2S_s$s.pt ] || env $E STRAT_QPHYS=2 STRAT_QSHAPE=1 STRAT_OUT_TAG=_QHAR2S QW_GPU_GB=30 runs/capped.sh qQHAR2S 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s qb >> $LOG 2>&1 || { echo "CRASH QHAR2S s$s $(date -Is)" >> $LOG; exit 1; }
done
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1AH_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2S_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh s42f5 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F5 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "S42_F5_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F5/diii_d_public_test.npz runs/sophelio_sota/submission_F4/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "S42_F5_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
set -a; . ./.env; set +a
curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/976250/" -o /dev/null -w "[f5] delete 976250 http %{http_code}\n" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F5/submission.zip F5 "F5: F4 + q95/betaN heads with plasma-shape features (out-of-fold shape-head kappa, delta, a, R_geo, q-shape term)" >> $LOG 2>&1
echo "S42_F5_DONE $(date -Is)" >> $LOG
