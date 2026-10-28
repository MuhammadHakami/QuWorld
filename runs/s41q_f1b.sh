#!/bin/bash
# Fusion §41q F1b public build: F1a + q/b from the all-data QHA heads (3 seeds) 50/50 with the ORIGINAL entry ensemble (the all-data entry
# has two diverged members, trunk1_t19-ew2 s1 / s2: in-sample q95 0.986 / 0.991 - no clip; refit for F2) (was: QW_ENTRY_DIR=
# trunks_all) + the J1A heads double-weighted (register heads J1QPIpi s1 / s2 at 1/7 instead of 2/7: F1a's consistency gain was half the
# projection). Gate: psi agreement with F1a >= 0.999; then upload. Expected public S ~ 0.9952-0.9953 (F1a 0.995143 + q/b data + less dilution).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41q_f1b.log
until [ -f $L/head_qb_QHA_s3.pt ]; do sleep 60; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1A_s$s+"; done)
H="${J}${J}head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2"
Q=$(for s in 1 2 3; do printf "head_qb_QHA_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
QW_GPU_GB=30 runs/capped.sh s41qb 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F1b --psi psiFA \
  --qb qh50 --qbheads "$Q" --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "S41Q_F1B_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F1b/diii_d_public_test.npz runs/sophelio_sota/submission_F1a/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "S41Q_F1B_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
runs/codabench_upload.sh runs/sophelio_sota/submission_F1b/submission.zip F1b "F1b: F1a + all-data q95/betaN heads (original entry), register heads at 1/7" >> $LOG 2>&1
echo "S41Q_F1B_DONE $(date -Is)" >> $LOG
