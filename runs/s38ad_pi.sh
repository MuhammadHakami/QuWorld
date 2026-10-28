#!/bin/bash
# doc 35 §38ad (physics arm): J1QPI with an active volume constraint (w_pi 0.1), all else as J1QPI -> isolates the physics term.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while ! grep -q "S38AD_DONE" $L/s38ad_fallback.log 2>/dev/null; do sleep 60; done
P='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
flock runs/strat/gpu.lock env QW_GPU_GB=60 runs/capped.sh s38ad 60 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --fixed "$P" --tag J1QPIpi \
  --epochs 60 --seed 1 >> $L/s38ad_pi.log 2>&1
.venv/bin/python paper/strat/fusion/head_screen.py head_boundary_v8_J1_s1 head_boundary_v8_J1QPI_s1 head_boundary_v8_J1QPIpi_s1 >> $L/s38ad_pi.log 2>&1
echo "S38AD_PI_DONE $(date -Is)" >> $L/s38ad_pi.log
