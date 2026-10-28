#!/bin/bash
# doc 35 §41f: PSI-TA (fixed-coordinate Thomson ADDED to the old block) seed 2 beat PSI-F seed 2 on VALID (0.999810 vs 0.999767);
# seeds 1 / 3, then the VALID-sample gate of HJl9 with psi from PSI-TA (tag HJl9TA; HJl9 sample official 0.99562).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "thz": true'
for s in 1 3; do
  QW_GPU_GB=40 runs/capped.sh psita$s 60 .venv/bin/python paper/sophelio/sota_train.py --tag s41f_PSI-TA-s$s --score-val 0 --cfg "{$CFG, \"seed\": $s}" > runs/strat/fusion/s41f_PSI-TA-s$s.log 2>&1
done
cd paper/strat/fusion
QW_GPU_GB=50 ../../../runs/capped.sh hjl9ta 60 ../../../.venv/bin/python gpu_build.py --split sample --tag HJl9TA --psi psiTA \
  --heads head_boundary_v8_J1_s1+head_boundary_v8_J1_s2+head_boundary_v8_J1_s3+head_boundary_v8_J1_s4+head_boundary_v8_J1_s5+head_boundary_v8_J1_s6 \
  --kw '{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}' --qb qh50 --axis head \
  --qbheads head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6 > ../../../runs/strat/fusion/gpu_sample_HJl9TA.log 2>&1
echo S41F_PSITA_DONE >> ../../../runs/strat/fusion/gpu_sample_HJl9TA.log
