#!/bin/bash
# Fusion §42b + §42c (registered before any result): psi trunks LS_w3 (level-set loss, training) and K128 / K256 (more PCA components),
# all PSI-F recipe seed 1 vs LS_ctrl; ONE VT read-out (raw + VB chain) for the four psi sets under runs/strat/vt.lock.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42bc.log
CFG='"width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 1, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"'
while pgrep -f "[s]ota_train.py --tag K(128|256)" > /dev/null; do sleep 60; done   # relaunch-safe: let an in-flight K trunk finish
for arm in "K128|128" "K256|256"; do
  n=${arm%%|*}; k=${arm#*|}
  [ -f runs/sophelio_sota/$n.pt ] || QW_GPU_GB=30 runs/capped.sh s42c 40 .venv/bin/python paper/sophelio/sota_train.py --tag $n --score-val 0 \
    --cfg "{\"k\": $k, $CFG}" >> $LOG 2>&1 || { echo "CRASH $n $(date -Is)" >> $LOG; exit 1; }
done
# 17:11: LS_w3 killed for memory (§42b reruns later); the read-out compares K128 / K256 with LS_ctrl.
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
VB='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42bcvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag BC \
  --psi psiF --psis LS_ctrl,K128,K256 --qb qh50 --qbheads "$Q" --heads "$J" --kws "{\"raw\": {\"_steps\": []}, \"VB\": $VB}" --axis head \
  >> $LOG 2>&1 || echo "CRASH vt BC $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py LS_ctrl_raw K128_raw K256_raw >> ../../../$LOG 2>&1
../../../.venv/bin/python vt_pp_compare.py LS_ctrl_VB K128_VB K256_VB >> ../../../$LOG 2>&1
echo "S42BC_DONE $(date -Is)" >> ../../../$LOG
