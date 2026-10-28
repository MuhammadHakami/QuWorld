#!/bin/bash
# doc 35 §40u: official validator (600 s, CUDA, Spark alone) on the real archive (cold_after 15000) and on a test copy
# (cold_after 300, cold_len 200) that exercises the cold portfolio under the official harness.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
V=baseline/Learn2Design-2026/tools/validate_submission.py; PY=baseline/Learn2Design-2026/.venv/bin/python
QW_GPU_GB=30 runs/capped.sh l2dval 30 $PY $V runs/l2d/final/submission_v6.zip --run-local --max-time-seconds 600 > runs/l2d/validate_v6.log 2>&1
QW_GPU_GB=30 runs/capped.sh l2dval 30 $PY $V /tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad/v6test/v6test.zip \
  --run-local --max-time-seconds 600 > runs/l2d/validate_v6test.log 2>&1
echo "S40U_VALIDATE_DONE $(date -Is)" >> runs/l2d/validate_v6test.log
