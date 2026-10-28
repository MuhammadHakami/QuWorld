#!/bin/bash
# Fusion §41q F2 public build: psiFA + heads J1AH x6 (all data + thz + full-rate history) double-weighted + J1QPIpi s1 / s2 (register, 1/7)
# + q/b qh50 with QHA x3 and the all-data entry (QW_ENTRY_DIR=trunks_all; t19-ew2 members refit with clip). Gate vs F1b (psi >= 0.999).
# Expected public S ~ 0.99523 (range 0.99515-0.9953), stated before the build.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41q_f2.log
until [ -f $L/head_boundary_v8_J1AH_s2.pt ]; do
  timeout 60 ssh -i ~/.ssh/id_ed25519_quworld merphistophline "grep -q HEAD_DONE ~/Documents/paper/quworld/runs/strat/fusion/J1AH_s2.log" 2>/dev/null \
    && timeout 900 rsync -a -e "ssh -i $HOME/.ssh/id_ed25519_quworld" merphistophline:Documents/paper/quworld/runs/strat/fusion/head_boundary_v8_J1AH_s2.{pt,npz} $L/
  sleep 60
done
until grep -q S41Q_F2HEADS_DONE $L/s41q_f2heads.log 2>/dev/null; do sleep 60; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1AH_s$s+"; done); H="${J}${J}head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2"
Q=$(for s in 1 2 3; do printf "head_qb_QHA_s$s+"; done | sed 's/+$//')
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
QW_ENTRY_DIR=trunks_all QW_GPU_GB=30 runs/capped.sh s41qf2b 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag F2 --psi psiFA \
  --qb qh50 --qbheads "$Q" --heads "$H" --kw "$KW" --axis head >> $LOG 2>&1 || { echo "S41Q_F2_FAIL build $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/agree.py runs/sophelio_sota/submission_F2/diii_d_public_test.npz runs/sophelio_sota/submission_F1b/diii_d_public_test.npz >> $LOG 2>&1 \
  || { echo "S41Q_F2_FAIL agreement $(date -Is)" >> $LOG; exit 1; }
runs/codabench_upload.sh runs/sophelio_sota/submission_F2/submission.zip F2 "F2: all-data psi (canonical) + all-data thz+history shape heads + register heads + all-data q/b heads and entry" >> $LOG 2>&1
echo "S41Q_F2_DONE $(date -Is)" >> $LOG
