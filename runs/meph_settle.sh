#!/bin/bash
# doc 35 §41p S1: does the v9b cold-run settle (lr 0.01 from the run's best at 60%, 0.003 at 85%) find deeper designs than plain
# NAAdamGD cold runs? R3-6 (PoolHop reached 0.157 there), 2 lanes, short warm phase, cold runs of 4,000 steps, 12,000 evaluations,
# paired (same seeds) with and without the settle.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
$PY paper/l2d/run_submission.py --round 3 --run 6 --sub paper/l2d/submission_v9 --evals 12000 --kw '{"lanes": 2, "cold_after": 1000, "cold_len": 4000}' --tag s1_settle > runs/l2d/sub_s1_settle_r3run6.log 2>&1
$PY paper/l2d/run_submission.py --round 3 --run 6 --sub paper/l2d/submission_v9 --evals 12000 --kw '{"lanes": 2, "cold_after": 1000, "cold_len": 4000, "cold_settle": []}' --tag s1_plain > runs/l2d/sub_s1_plain_r3run6.log 2>&1
