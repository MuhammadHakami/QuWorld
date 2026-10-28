#!/bin/bash
# mephiles: official validator on CUDA (alone on the GPU) for the v3b zip (doc 35 §40n), 600 s.
cd ~/Documents/paper/quworld/baseline/Learn2Design-2026 && XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python tools/validate_submission.py \
  ../../runs/l2d/final/submission_v3b.zip --run-local --max-time-seconds 600 > ../../runs/l2d/final/validate_cuda_v3b.log 2>&1
echo "VAL_CUDA_DONE $(date -Is)" >> ../../runs/l2d/final/validate_cuda_v3b.log
