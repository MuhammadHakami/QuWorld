#!/bin/bash
# §38g corrected inputs under the head pipeline (VALID sample only)
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
flock runs/strat/gpu.lock env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=3 \
  STRAT_INPUTS=magfix STRAT_OUT_TAG=_G1 QW_GPU_GB=24 runs/capped.sh s38g 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > $L/s38g_G1.log 2>&1
flock runs/strat/gpu.lock env QW_GPU_GB=24 runs/capped.sh s38g 40 .venv/bin/python paper/sophelio/sota_train.py --tag s38g_QF --score-val 0 \
  --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 30, "w_scal": 1.5123263659215784, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 2, "edge_w": 2.0, "fast": true, "inputs": "magfix"}' > $L/s38g_QF.log 2>&1
flock runs/strat/gpu.lock env QW_GPU_GB=24 runs/capped.sh s38g 40 .venv/bin/python paper/sophelio/sota_train.py --tag s38g_QC --score-val 0 \
  --cfg '{"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 30, "w_scal": 1.5123263659215784, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "seed": 2, "edge_w": 2.0, "fast": true}' > $L/s38g_QC.log 2>&1
.venv/bin/python paper/strat/fusion/head_screen.py head_boundary_v8_H1_s1 head_boundary_v8_G1_s1 > $L/s38g_heads.log 2>&1
.venv/bin/python paper/strat/fusion/qb_eval.py ../nas/sophelio/trunks/trunk1_t19-ew2_seed2.pt s38g_QC.pt s38g_QF.pt s38g_QF.pt@entry s38g_QC.pt@entry > $L/s38g_qb.log 2>&1
echo S38G_DONE >> $L/s38g_qb.log
