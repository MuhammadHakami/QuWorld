#!/bin/bash
# merphistophline: §40r v4 runs 5 and 8 (extra topologies for the screen; non-finite counts saved).
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
for r in 5 8; do $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v4 --kw '{"loglen": true}' > runs/l2d/sub_v4_run$r.log 2>&1; done
echo "L2D_RQ14_DONE $(date -Is)" >> runs/l2d/rq14.log
