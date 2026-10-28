#!/bin/bash
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
for seed in 1 3; do
  flock runs/strat/gpu.lock env QW_GPU_GB=24 runs/capped.sh s38gq 40 .venv/bin/python paper/sophelio/sota_train.py --tag s38g_QF-s$seed --score-val 0 \
    --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 30, "w_scal": 1.5123263659215784, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "fast": true, "inputs": "magfix", "seed": '$seed'}' > runs/strat/fusion/s38g_QF-s$seed.log 2>&1
done
STRAT_TAG=_3seed .venv/bin/python paper/strat/fusion/qb_eval.py s38g_QF.pt+s38g_QF-s1.pt+s38g_QF-s3.pt s38g_QF.pt+s38g_QF-s1.pt+s38g_QF-s3.pt@entry > runs/strat/fusion/s38g_qb3.log 2>&1
echo S38GQ_DONE >> runs/strat/fusion/s38g_qb3.log
