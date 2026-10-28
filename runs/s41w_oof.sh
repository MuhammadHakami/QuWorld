#!/bin/bash
# Fusion §41w: thp re-test with OUT-OF-FOLD psi (the §41r failure = in-sample psi on the training rows). PSIO_k0..4 = PSI-F config leaving
# out stratified TRAIN fold k (cv_shots, as the head CV); thp_all_psiOOF: TRAIN rows from the trunk that never saw their fold, VALID /
# TEST rows from the mean of the 5. Then proxy CV P5thpo vs P5thf (5 folds, all TRAIN) and full-size J1THPO_s1 vs J1THS_s1 on VT.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41w_oof.log
until grep -q S41T_SCREEN_DONE $L/s41t_screen.log 2>/dev/null; do sleep 30; done
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"'
for k in 0 1 2 3 4; do
  [ -f runs/sophelio_sota/PSIO_k$k.pt ] || flock runs/strat/gpu.lock env QW_GPU_GB=30 runs/capped.sh s41wpsi 40 .venv/bin/python paper/sophelio/sota_train.py \
    --tag PSIO_k$k --score-val 0 --cfg "{$CFG, \"cv\": \"$k/5\"}" >> $LOG 2>&1 || { echo "CRASH PSIO k$k" >> $LOG; exit 1; }
done
QW_GPU_GB=20 runs/capped.sh s41wthp 40 .venv/bin/python paper/strat/fusion/thp_feats.py feat --psi psiOOF >> $LOG 2>&1 || { echo "CRASH thp psiOOF" >> $LOG; exit 1; }
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_INPUTS=magfix STRAT_THZ=1 STRAT_HFULL=1"
for k in 0 1 2 3 4; do
  [ -f $L/head_boundary_v8_P5thpoF${k}_s1_oof.npz ] || env $E STRAT_DEPTH=2 STRAT_WIDTH=1024 STRAT_EPOCHS=30 STRAT_THP=psiOOF STRAT_CV=$k/5 STRAT_OUT_TAG=_P5thpoF$k \
    QW_GPU_GB=16 runs/capped.sh s41wcv 30 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/P5thpoF$k.log 2>&1 || echo "CRASH P5thpo F$k" >> $LOG
done
CV_PFX=P5 .venv/bin/python paper/strat/fusion/cv_report.py thf thpo >> $LOG 2>&1
[ -f $L/head_boundary_v8_J1THPO_s1.pt ] || flock runs/strat/gpu.lock env $E STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 QW_MAGFIX=X_magfix STRAT_THP=psiOOF \
  STRAT_OUT_TAG=_J1THPO QW_GPU_GB=40 runs/capped.sh s41wJ 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 >> $LOG 2>&1 || echo "CRASH J1THPO" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1THS_s1 head_boundary_v8_J1THPO_s1 >> $LOG 2>&1 || echo "CRASH vt" >> $LOG
echo "S41W_DONE $(date -Is)" >> $LOG
