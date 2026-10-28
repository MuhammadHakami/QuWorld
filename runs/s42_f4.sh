#!/bin/bash
# Fusion F4 public build (registered 2026-10-12T01:06, before the §42m result): F3 with the psi trunks replaced by the §42m winner
#   (PQ2~g -> psiPQ2A = PQ2A s1-2 [s3 when present]; PQmix~g -> psiMixA = PQ2A s1-2 + psiFA s1-3) + axis guard (fallback psiFA, 0.25 m) + the winning
#   chain (VB or ns), q/b heads QHAR2 s1-3 (F3: s1-2) at the head weight in qb_blend_w.txt. Skip if no §42m arm passes.
# Expected public S ~ 0.99572 (range 0.99560-0.99590) vs F3 0.995565. Stated before the build.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42_f4.log
until grep -q "^S42M_VT_DONE" $L/s42m_vt.log 2>/dev/null; do sleep 60; done
pick=$(python3 - $L/s42m_vt.log <<'PY'
import re, sys
best, arm = 0.0, ""
for t, d, lo, dl in re.findall(r"\[vtpp\] (\S+) vs LS_ctrl_VB: .*?delta ([-+0-9.]+) \[([-+0-9.]+), [-+0-9.]+\] \| .*?dlcfs ([-+0-9.]+)", open(sys.argv[1]).read()):
    if "~g_" in t and float(lo) > 0 and float(dl) <= 0 and float(d) > best:
        best, arm = float(d), t
print(arm)
PY
)
echo "[f4] §42m pick: ${pick:-none} $(date -Is)" >> $LOG
[ -n "$pick" ] || { echo "S42_F4_SKIP no passing arm $(date -Is)" >> $LOG; exit 0; }
case "$pick" in PQmix*) PS=psiMixA ;; *) PS=psiPQ2A ;; esac
case "$pick" in
  *_ns) KW='{"_steps":["bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' ;;
  *)    KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' ;;
esac
until grep -q "^PQ2A_s1 " $L/s42m.log 2>/dev/null; do sleep 60; done                      # 01:23 (after the OOM reboot): PQ2A s1-2 (speed; s3 for later builds)
for s in 1 2; do [ -f runs/sophelio_sota/PQ2A_s$s.pt ] || { echo "S42_F4_FAIL missing PQ2A_s$s $(date -Is)" >> $LOG; exit 1; }; done
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.5)
echo "[f4] psi $PS, chain ${pick##*_}, q/b head weight $QBW $(date -Is)" >> $LOG
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1AH_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//')
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh s42f4 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F4 --psi $PS \
  --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "S42_F4_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F4/diii_d_public_test.npz runs/sophelio_sota/submission_F3/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "S42_F4_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
set -a; . ./.env; set +a
curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/976044/" -o /dev/null -w "[f4] delete 976044 http %{http_code}\n" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F4/submission.zip F4 "F4: all-data psi trunks with physics-scaling + coil/Ip-ratio inputs (canonical currents) + axis guard + 8 all-data thz+history shape heads + register heads (PI-KAN + 30-atom Rydberg layer) + all-data q/b heads with physics scalings (3 seeds) and entry; post-processing chain ${pick##*_}" >> $LOG 2>&1
echo "S42_F4_DONE $(date -Is)" >> $LOG
