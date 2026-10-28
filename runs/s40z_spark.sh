#!/bin/bash
# doc 35 §40z restart distribution on the Spark (one process at a time; K 8 population at XLA fraction 0.45): 8 independent
# noisy-NAAdamGD + barrier chains (no selection), 1,200 steps each, from cold random starts, Hamming-nearest dataset donors,
# or the dataset's best designs (good family, loss < -0.1, transplanted by label); per-chain best feasible losses ->
# probability that one chain escapes the 0.42 plateau -> projected best of ~100 chains in 4 h.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "pop_probe[.]py" > /dev/null; do sleep 60; done
for job in "1 best" "8 best" "1 donor" "6 cold" "6 donor" "8 cold" "9 cold" "3 cold" "10 cold"; do set -- $job
  QW_GPU_GB=60 runs/capped.sh l2drd$1$2 60 $PY paper/l2d/pop_probe.py --round 3 --run $1 --arm naadam --k 8 --replace 0 --s 300 --rounds 4 \
    --starts $2 --tag rd_$2 > runs/l2d/pop_rd_$2_r3run$1.log 2>&1
done
