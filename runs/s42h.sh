#!/bin/bash
# Fusion §42h (registered before any result): PQ2 trunk (PSI-F recipe + qphys2) vs LS_ctrl on all VT shots.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42h.log
until [ $(grep -c "^\[qbw\]" $L/qb_blend_vt.log 2>/dev/null) -ge 4 ] || grep -q Traceback $L/qb_blend_vt.log 2>/dev/null; do sleep 60; done
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "qphys": 2'
[ -f runs/sophelio_sota/PQ2.pt ] || QW_GPU_GB=30 runs/capped.sh s42h 40 .venv/bin/python paper/sophelio/sota_train.py --tag PQ2 --score-val 0 \
  --cfg "{$CFG}" >> $LOG 2>&1 || { echo "CRASH PQ2 $(date -Is)" >> $LOG; exit 1; }
until grep -q "^S42BC_DONE" $L/s42bc.log 2>/dev/null; do sleep 60; done
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
VB='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42hvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag H \
  --psi psiF --psis PQ2 --qb qh50 --qbheads "$Q" --heads "$J" --kws "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" --axis head >> $LOG 2>&1 || echo "CRASH vt H $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py LS_ctrl_raw PQ2_raw >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py LS_ctrl_VB PQ2_VB >> ../../../$LOG 2>&1
echo "S42H_DONE $(date -Is)" >> ../../../$LOG
