#!/bin/bash
# F13 (§43l pass, registered 2026-10-14T23:10): F8's exact configuration with the three TRAIN-only J1QH register heads ADDED to the shape
# ensemble at the VT-validated proportions (A = 6 J1RL : 1 J1QPIpi : 3 J1QH -> production 24 J1RLA : 4 J1QPIpi : 12 J1QH, i.e. x4 each).
# Builds ALONE (vt.lock), agreement vs F8, then F13K = post_kw2 (as F8K / F12K). No upload here.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/f13.log
KW=$(grep -o '"_steps".*' $L/f8.log | head -1 | sed 's/ 20[0-9][0-9]-.*$//'); KW="{$KW"
QBW=$(cat $L/qb_blend_w.txt 2>/dev/null || echo 0.8)
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLA_s$s*3+"; done)"head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2_s$s+"; done | sed 's/+$//'); XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"
echo "[f13] heads $H kw $KW qbw $QBW $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh f13 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F13 \
  --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw $QBW --qbheads "$Q" --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp >> $LOG 2>&1 || { echo "F13_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F13/diii_d_public_test.npz runs/sophelio_sota/submission_F8/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "F13_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
echo "F13_BUILT $(date -Is)" >> $LOG
QW_GPU_GB=12 runs/capped.sh kw2f13 30 .venv/bin/python paper/strat/fusion/post_kw2.py --src F13 --tag F13K --heads "$H" >> $LOG 2>&1 && echo "F13K_DONE $(date -Is)" >> $LOG || echo "CRASH F13K $(date -Is)" >> $LOG
