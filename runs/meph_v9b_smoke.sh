#!/bin/bash
# v9b smoke on mephiles after G2: 2 lanes, tiny warm phase, cold runs of 40 steps with settle at 60% / 85%.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
while pgrep -f "[g]ood_start_probe" > /dev/null; do sleep 30; done
baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --round 3 --run 1 --sub paper/l2d/submission_v9 --evals 200 \
  --kw '{"lanes": 2, "k": 4, "k_wide": 4, "cold_after": 20, "cold_len": 40}' --tag v9bsmoke > runs/l2d/sub_v9bsmoke.log 2>&1
