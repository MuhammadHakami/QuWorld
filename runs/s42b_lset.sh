#!/bin/bash
# Fusion §42b (registered before any result): level-set LCFS loss trunks LS_ctrl / LS_w3 / LS_w30 (PSI-F recipe, TRAIN only, seed 1),
# then raw + VB-chain parts on all VT shots and paired comparisons vs LS_ctrl. Starts after §41y-b's VT run.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42b.log
# 16:30 relaunch: VT read-outs one at a time under runs/strat/vt.lock (the 16:20 memguard kill); LS_ctrl already trained.
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"'
for arm in "LS_ctrl|0" "LS_w3|3" "LS_w30|30"; do
  n=${arm%%|*}; w=${arm#*|}
  [ -f runs/sophelio_sota/$n.pt ] || QW_GPU_GB=30 runs/capped.sh s42b 40 .venv/bin/python paper/sophelio/sota_train.py --tag $n --score-val 0 \
    --cfg "{$CFG, \"lset_w\": $w}" >> $LOG 2>&1 || { echo "CRASH $n $(date -Is)" >> $LOG; exit 1; }
done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
VB='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
for n in LS_ctrl LS_w3 LS_w30; do
  flock runs/strat/vt.lock env QW_GPU_GB=20 runs/capped.sh s42bvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag $n --psi $n --qb qh50 --qbheads "$Q" \
    --heads "$J" --kws "{\"${n}_raw\": {\"_steps\": []}, \"${n}_VB\": $VB}" --axis head >> $LOG 2>&1 || echo "CRASH vt $n" >> $LOG
done
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py LS_ctrl_raw LS_w3_raw LS_w30_raw >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py LS_ctrl_VB LS_w3_VB LS_w30_VB >> ../../../$LOG 2>&1
echo "S42B_DONE $(date -Is)" >> ../../../$LOG
