#!/bin/bash
# doc 35 §41o G2: G1 (the incumbent noisy chain throws good designs out: -0.207 -> +3.1 -> plateau +0.39) repeated WITHOUT the
# exploration noise (Adam lr 0.1) and with lr 0.01: is the noise or the Adam step the ejector? If noise-free Adam holds good
# designs, a long noisy phase followed by a noise-free settle (the #2 / #5 late drop) is the mechanism to copy.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=25 runs/capped.sh g2 25 $PY paper/l2d/good_start_probe.py --k 2 --steps 1000 --no-noise --tag g2 > runs/l2d/good_start_g2.log 2>&1
QW_GPU_GB=25 runs/capped.sh g2b 25 $PY paper/l2d/good_start_probe.py --k 2 --steps 1000 --no-noise --lr 0.01 --tag g2lr > runs/l2d/good_start_g2lr.log 2>&1
