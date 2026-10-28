#!/bin/bash
# F5c (registered 2026-10-12 after the §42p / §42s results; POST HOC, labelled as such in doc 35): combine the two consistent near-misses.
# Shape heads J1RLA s1-8 (J1AH recipe + qphys2 + liphys, all data; §42p two seeds +0.000374 / +0.000395 mean-7) replace J1AH s1-8;
# q/b heads QHAR2S s1-3 (QHAR2 + qshape, all data; §42s two seeds +0.000185 / +0.000159) replace QHAR2 s1-3. Everything else = F4.
# Expected public = 0.995575 + 0.20 x 0.000385 (cons, carry ~1) + 0.225 x 0.000172 (q/b) ~ 0.99569 (range 0.99560-0.99575).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f5c.log
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_ALL=1 QW_MAGFIX=X_magfix"
echo "F5C_START $(date -Is)" >> $LOG
for s in 1 2 3; do
  [ -f $L/head_qb_QHAR2S_s$s.pt ] || env $E STRAT_QPHYS=2 STRAT_QSHAPE=1 STRAT_OUT_TAG=_QHAR2S QW_GPU_GB=30 runs/capped.sh qQHAR2S 40 \
    .venv/bin/python paper/strat/fusion/head_boundary.py $s qb >> $LOG 2>&1 || { echo "CRASH QHAR2S s$s $(date -Is)" >> $LOG; exit 1; }
  echo "QHAR2S_s${s}_TRAINED $(date -Is)" >> $LOG
done
for s in 1 2 3 4 5 6 7 8; do
  [ -f $L/head_boundary_v8_J1RLA_s$s.pt ] || env $E STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_J1RLA QW_GPU_GB=30 \
    runs/capped.sh hJ1RLA 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s >> $LOG 2>&1 || { echo "CRASH J1RLA s$s $(date -Is)" >> $LOG; exit 1; }
  echo "J1RLA_s${s}_TRAINED $(date -Is)" >> $LOG
done
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2S_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f5c 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F5c \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "F5C_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F5c/diii_d_public_test.npz runs/sophelio_sota/submission_F4/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F5C_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
set -a; . ./.env; set +a
curl -s --max-time 120 -X DELETE -H "Authorization: Token $CODABENCH_TOKEN" "https://www.codabench.org/api/submissions/976250/" -o /dev/null -w "[f5c] delete 976250 http %{http_code}\n" >> $LOG
runs/codabench_upload.sh runs/sophelio_sota/submission_F5c/submission.zip F5c "F5c: F4 + shape heads with Te^1.5 current-profile proxies + q95/betaN heads with out-of-fold plasma-shape features" >> $LOG 2>&1
echo "F5C_DONE $(date -Is)" >> $LOG
