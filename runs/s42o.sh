#!/bin/bash
# Fusion §42o (registered before any result): PQ2H = PQ2 + full-rate current history (hfull), TRAIN only, seed 1; VT vs PQ2 (both guarded).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42o.log
until grep -q "^PQL10_TRAINED\|^CRASH" $L/s42n.log 2>/dev/null; do sleep 30; done          # at most 3 GPU jobs
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "qphys": 2, "hfull": true'
[ -f runs/sophelio_sota/PQ2H.pt ] || QW_GPU_GB=30 runs/capped.sh s42o 40 .venv/bin/python paper/sophelio/sota_train.py --tag PQ2H --score-val 0 \
  --cfg "{$CFG}" >> $LOG 2>&1 || { echo "CRASH PQ2H $(date -Is)" >> $LOG; exit 1; }
echo "PQ2H_TRAINED $(date -Is)" >> $LOG
until grep -q "^S42N_DONE" $L/s42n.log 2>/dev/null; do sleep 30; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
VB='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42ovt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag O \
  --psi psiF --psis "PQ2H~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$J" --kws "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" --axis head >> $LOG 2>&1 || echo "CRASH vt O $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQ2~g_VB" "PQ2H~g_VB" >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py PQ2_raw "PQ2H~g_raw" >> ../../../$LOG 2>&1
echo "S42O_DONE $(date -Is)" >> ../../../$LOG
