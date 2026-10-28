#!/bin/bash
# Fusion §41q F1a interim public build (user: faster iterations): psiFA (all data, canonical) + heads J1A s1-5 (all data + thz) +
# J1QPIpi s1 / s2 (existing 30-atom register heads: hybrid) + q/b qh50 (original QH heads + entry) + HJl9 post-processing.
# Pre-upload gate: agree.py vs the HJl9 public npz (psi agreement >= 0.999, all finite), then upload (expected public S ~ 0.9953).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41q_f1a.log
until [ -f $L/head_boundary_v8_J1A_s5.pt ]; do sleep 60; done; sleep 30
H="$(for s in 1 2 3 4 5; do printf "head_boundary_v8_J1A_s$s+"; done)head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2"
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
QW_GPU_GB=30 runs/capped.sh s41qa 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F1a --psi psiFA --qb qh50 \
  --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "S41Q_F1A_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F1a/diii_d_public_test.npz runs/sophelio_sota/submission_HJl9/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "S41Q_F1A_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
runs/codabench_upload.sh runs/sophelio_sota/submission_F1a/submission.zip F1a "F1a: all-data psi (Ip-sign canonical) + all-data thz shape heads + register heads + HJl9 post" >> $LOG 2>&1
echo "S41Q_F1A_DONE $(date -Is)" >> $LOG
