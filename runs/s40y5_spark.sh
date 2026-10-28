#!/bin/bash
# doc 35 §40y on the Spark: rung 3 (v7a vs v6 end to end, 3k evaluations) first, then rung 2b (replaces the remaining P1 topologies): P2 staged population - one noisy NAAdamGD + barrier
# chain descends 400 steps, then K 8 branch from its best point (family-directed moves, child noise 0.1, 50-step rounds,
# Boltzmann selection, REINFORCE move policy; 2,400 chain-steps) vs the control continuing the single chain for the same
# steps; R3-1, R3-8, R3-6, one topology at a time (population at XLA fraction 0.4 + control 0.15). Then rung 1b.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "pop_probe[.]py" > /dev/null; do sleep 60; done
# rung 3 first (decides the submission): v7a (v6 + smooth barrier) and v6, end to end, 3,000 evaluations, R3-1 / R3-8 / R3-6
for sub in submission_v7 submission; do
  for r in 1 8 6; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 QW_GPU_GB=25 runs/capped.sh l2dsub${r} 25 $PY paper/l2d/run_submission.py --round 3 --run $r --evals 3000 \
      --sub paper/l2d/$sub --tag ${sub}_e3k > runs/l2d/sub_${sub}_e3k_r3run$r.log 2>&1 &
    sleep 90
  done
  wait
done
for r in 1 8 6; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 QW_GPU_GB=50 runs/capped.sh l2dp2k8r$r 50 $PY paper/l2d/pop_probe.py --round 3 --run $r --arm naadam \
    --warm_steps 400 --k 8 --s 50 --rounds 6 --child_s0 0.1 --tag p2k8 > runs/l2d/pop_p2k8_r3run$r.log 2>&1 &
  sleep 120
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 QW_GPU_GB=20 runs/capped.sh l2dp2k1r$r 20 $PY paper/l2d/pop_probe.py --round 3 --run $r --arm naadam \
    --warm_steps 400 --k 1 --replace 0 --s 50 --rounds 48 --tag p2k1 > runs/l2d/pop_p2k1_r3run$r.log 2>&1 &
  wait
done
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
for r in 1 8 6; do
  QW_GPU_GB=25 runs/capped.sh l2do1br$r 25 $PY paper/l2d/o1_race.py --round 3 --run $r --units 380 \
    --arms n-cadam+b n-ademamix+b n-soap+b n-adopt+b > runs/l2d/o1b_spark_r3run$r.log 2>&1 &
  sleep 90
done
wait
