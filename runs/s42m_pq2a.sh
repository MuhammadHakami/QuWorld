#!/bin/bash
# Fusion speed: train PQ2A s2/s1/s3 (psiFA recipe + qphys2, all data) NOW, in parallel with the VT queue, so a §42l pass builds F4 at
# once. Speculative production training (no rule depends on it); s42h_promote.sh skips seeds whose checkpoints exist.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42m.log
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "all_data": true, "canon": true, "qphys": 2'
for s in 2 1 3; do
  [ -f runs/sophelio_sota/PQ2A_s$s.pt ] || QW_GPU_GB=30 runs/capped.sh s42m 40 .venv/bin/python paper/sophelio/sota_train.py --tag PQ2A_s$s --score-val 0 \
    --cfg "{$CFG, \"seed\": $s}" >> $LOG 2>&1 || echo "CRASH PQ2A s$s $(date -Is)" >> $LOG
  echo "PQ2A_s$s $(date -Is)" >> $LOG
done
echo "S42M_DONE $(date -Is)" >> $LOG
