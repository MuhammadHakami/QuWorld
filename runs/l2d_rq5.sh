#!/bin/bash
# merphistophline (one ROCm process at a time): §40i plateau-escape probe on run 1, after rq4 (v1b run 6) finishes.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm
while pgrep -f "[l]2d_rq4.sh|[r]un_submission.py|[e]val_cands.py|[v]alidate_submission.py|[p]lateau_probe.py" > /dev/null; do sleep 30; done
baseline/Learn2Design-2026/.venv-rocm/bin/python paper/l2d/plateau_probe.py --run 1 --warm 2000 --branch 600 \
  --arms a0.1,a0.03,a0.01,a0.003,hop,shop > runs/l2d/plateau_run1.log 2>&1
