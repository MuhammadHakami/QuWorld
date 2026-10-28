#!/bin/bash
# doc 35 §41k BH1 on mephiles (queued behind runs/meph_l2.sh): basin hopping, structural moves + L-BFGS repair, R3-1 then R3-8.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "[m]eph_l2.sh" > /dev/null; do sleep 30; done
$PY paper/l2d/bh_probe.py --round 3 --run 1 --moves 30 --repair 150 > runs/l2d/bh_bh1_r3run1.log 2>&1
$PY paper/l2d/bh_probe.py --round 3 --run 8 --moves 30 --repair 150 > runs/l2d/bh_bh1_r3run8.log 2>&1
