#!/bin/bash
# merphistophline: §40s v5 (xcell) runs 5 and 8 (extra topologies), after v4 run 8.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
while pgrep -f "[l]2d_rq14.sh" > /dev/null || pgrep -f "[r]un_submission.py" > /dev/null; do sleep 30; done
for r in 5 8; do $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v5 --kw '{"hop_mode": "xcell", "patience": 500, "seg": 150, "lr_hop": 0.01, "pool_every": 30}' > runs/l2d/sub_v5x_run$r.log 2>&1; done
echo "L2D_RQ16_DONE $(date -Is)" >> runs/l2d/rq16.log
