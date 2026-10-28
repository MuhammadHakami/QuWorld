#!/bin/bash
# F15 (§45l): waits for J1RLHA s1-8 + QHAR2Hu s1-3, builds F15 exactly as F14 (vt.lock, li tree w 0.2, same weights / kw / qbw / xp heads) with
# the Huber heads swapped in, agreement vs F14, F15K = post_kw2, then uploads F15K (expected public ~0.99593, range 0.99585-0.99600; stated
# in doc 35 §45l before upload). Storage: on a dataset-create failure, deletes the superseded F12K submission (986792) and retries once.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 QW_LITREE=/home/mohhakami/Documents/paper/quworld/runs/strat/fusion/litree.npz:0.2
L=runs/strat/fusion; LOG=$L/f15.log
until grep -q "^F15_SHAPE_DONE" $LOG && grep -q "^F15_QB_DONE" $LOG; do grep -q "^CRASH" $LOG && exit 1; sleep 60; done
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLHA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2Hu_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
echo "[f15] heads $H qb $Q kw $KW qbw $QBW $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f15 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F15 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F15_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F15/diii_d_public_test.npz runs/sophelio_sota/submission_F14/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F15_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F15_BUILT $(date -Is)" >> $LOG
QW_GPU_GB=12 runs/capped.sh kw2f15 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F15 --tag F15K --heads "$H" >> $LOG 2>&1 || { echo "CRASH F15K $(date -Is)" >> $LOG; exit 1; }
echo "F15K_DONE $(date -Is)" >> $LOG
M="F15K: F14K with Huber-loss shape heads (J1RLHA) and q/b heads (QHAR2Hu)"
runs/codabench_upload.sh runs/sophelio_sota/submission_F15K/submission.zip F15K "$M" >> $LOG 2>&1
if grep -q "dataset create failed" runs/sophelio_sota/upload_F15K.log; then
  set -a; . ./.env; set +a
  curl -s -o /dev/null -w "%{http_code}" --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/986792/" > $L/f15_del.txt
  echo "[retry] after deleting superseded F12K (986792) http $(cat $L/f15_del.txt) $(date -Is)" >> $LOG
  runs/codabench_upload.sh runs/sophelio_sota/submission_F15K/submission.zip F15K "$M" >> $LOG 2>&1
fi
id=$(grep -o '"id":[0-9]*' runs/sophelio_sota/upload_F15K.log | head -1 | cut -d: -f2); echo "[upload] F15K id $id $(date -Is)" >> $LOG
for i in $(seq 1 120); do
  s=$(runs/codabench_status.sh $id 2>/dev/null); case "$s" in Finished*|Failed*) echo "[score] F15K $id $s $(date -Is)" >> $LOG; break;; esac; sleep 60
done
echo "UPLOAD_F15K_DONE $(date -Is)" >> $LOG
