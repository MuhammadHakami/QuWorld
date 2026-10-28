#!/bin/bash
# doc 35 §41a D1: single-tuning detuning sweeps (16 shifts x every tuning) + best pairs from our plateau designs R3-1 / R3-8.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
PY=baseline/Learn2Design-2026/.venv/bin/python
for r in 1 8; do
  QW_GPU_GB=18 runs/capped.sh d1r$r 18 $PY paper/l2d/detune_probe.py --round 3 --run $r --hist runs/l2d/hist_submission_v7_e3k_run$r.npz > runs/l2d/detune_d1_r3run$r.log 2>&1
done
