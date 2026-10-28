#!/bin/bash
# Fusion §42n (registered before any result): PQL10 = PQ2 + level-set LCFS loss (lset_w 10), TRAIN only, seed 1; VT vs PQ2 (both guarded).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42n.log
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "qphys": 2, "lset_w": 10'
[ -f runs/sophelio_sota/PQL10.pt ] || QW_GPU_GB=30 runs/capped.sh s42n 40 .venv/bin/python paper/sophelio/sota_train.py --tag PQL10 --score-val 0 \
  --cfg "{$CFG}" >> $LOG 2>&1 || { echo "CRASH PQL10 $(date -Is)" >> $LOG; exit 1; }
echo "PQL10_TRAINED $(date -Is)" >> $LOG
until grep -q "^S42M_VT_DONE" $L/s42m_vt.log 2>/dev/null; do sleep 30; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
VB='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42nvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag N \
  --psi psiF --psis "PQL10~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$J" --kws "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" --axis head >> $LOG 2>&1 || echo "CRASH vt N $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQ2~g_VB" "PQL10~g_VB" >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py PQ2_raw "PQL10~g_raw" >> ../../../$LOG 2>&1
echo "S42N_DONE $(date -Is)" >> ../../../$LOG
