#!/bin/bash
# doc 35 §41l (Fusion, full data): PSI-H = PSI-F config + long-window input history, seed 2 (vs PSI-F seed 2 VALID 0.999767);
# PSI-F seeds 4 / 5 / 6 for a 6-seed psi ensemble (gate psiF6 vs HJl9's 3-seed psiF on the official VALID sample).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix"'
QW_GPU_GB=40 runs/capped.sh psih 60 .venv/bin/python paper/sophelio/sota_train.py --tag s41l_PSI-H --score-val 0 --cfg "{$CFG, \"hist\": true, \"seed\": 2}" > runs/strat/fusion/s41l_PSI-H.log 2>&1
for s in 4 5 6; do
  QW_GPU_GB=40 runs/capped.sh psif$s 60 .venv/bin/python paper/sophelio/sota_train.py --tag s38i_PSI-F-s$s --score-val 0 --cfg "{$CFG, \"seed\": $s}" > runs/strat/fusion/s38i_PSI-F-s$s.log 2>&1
done
cd paper/strat/fusion
QW_GPU_GB=50 ../../../runs/capped.sh hjl9f6 60 ../../../.venv/bin/python gpu_build.py --split sample --tag HJl9F6 --psi psiF6 \
  --heads head_boundary_v8_J1_s1+head_boundary_v8_J1_s2+head_boundary_v8_J1_s3+head_boundary_v8_J1_s4+head_boundary_v8_J1_s5+head_boundary_v8_J1_s6 \
  --kw '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' --qb qh50 --axis head \
  --qbheads head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6 > ../../../runs/strat/fusion/gpu_sample_HJl9F6.log 2>&1
echo S41L_DONE >> ../../../runs/strat/fusion/gpu_sample_HJl9F6.log
