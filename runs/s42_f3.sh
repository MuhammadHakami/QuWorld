#!/bin/bash
# Fusion §42 F3 public build (re-registered 2026-10-11T17:43, after §42a passed; before any F3 result):
#   psiFA + J1AH s1-8 (weight 3 each) + register heads J1QPIpi s1-2 (weight 2 each; register share 4/28 = F2's 1/7)
#   + q/b qh-blend with QHAR2 s1-2 (all-data q/b heads + §42e qphys2: scalings + coil / Ip ratios) at the head weight in runs/strat/fusion/qb_blend_w.txt
#   (default 0.5; re-tuned on VT) + the all-data entry; post-processing = HJl9 chain, + the §41y-b arm that passes its rule (none if the
#   §41y-b read-out failed to run).
# Expected public S ~ 0.99550 (range 0.99540-0.99565); F2 = 0.99539 (q/b 0.99598 -> ~0.9965). Stated before the build.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42_f3.log
until [ -f $L/head_qb_QHAR2_s1.pt ] && [ -f $L/head_qb_QHAR2_s2.pt ]; do sleep 120; done; sleep 30   # speed: 2 seeds (s3 for later builds)
# 20:12: F3 no longer waits for §41y-b (slow read-out, ~1e-4 lever): use its arm only if the verdict exists at build time.
pick=$(python3 - $L/s41y_vt3.log <<'PY'
import re, sys
best, arm = 0.0, ""
for t, d, lo, dl in re.findall(r"\[vtpp\] (\S+) vs VB: .*?delta ([-+0-9.]+) \[([-+0-9.]+), [-+0-9.]+\] \| .*?dlcfs ([-+0-9.]+)", open(sys.argv[1]).read()):
    if float(lo) > 0 and float(dl) <= 0 and float(d) > best:
        best, arm = float(d), t
print(arm)
PY
)
case "$pick" in
  VL2) KW='{"_steps":["shift","bw","kw","xp","jgn","li","li2"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' ;;
  VX)  KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10,"n_it":6,"dk":0.006},"li":{"cap":0.9}}' ;;
  *)   KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' ;;
esac
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.5)
echo "[f3] post-processing arm: ${pick:-none (HJl9 chain)}; q/b head weight $QBW $(date -Is)" >> $LOG
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1AH_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//')
QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh s42f3 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F3 --psi psiFA \
  --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "S42_F3_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F3/diii_d_public_test.npz runs/sophelio_sota/submission_F2/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "S42_F3_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
set -a; . ./.env; set +a
curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/960275/" -o /dev/null -w "[f3] delete 960275 http %{http_code}\n" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F3/submission.zip F3 "F3: all-data psi (canonical) + 8 all-data thz+history shape heads + register heads (PI-KAN + 30-atom Rydberg layer) + all-data q/b heads with physics scalings (Bt/Ip, pressure moments, coil/Ip ratios) and entry; post-processing ${pick:-HJl9}" >> $LOG 2>&1
echo "S42_F3_DONE $(date -Is)" >> $LOG
