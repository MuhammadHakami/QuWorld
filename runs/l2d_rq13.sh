#!/bin/bash
# merphistophline: §40r v4 runs 4 and 6 (ROCm; non-finite counts saved), after §40q.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
while pgrep -f "[l]2d_rq12.sh" > /dev/null || pgrep -f "[d]onor_scan.py" > /dev/null; do sleep 30; done
for r in 4 6; do $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v4 --kw '{"loglen": true}' > runs/l2d/sub_v4_run$r.log 2>&1; done
echo "L2D_RQ13_DONE $(date -Is)" >> runs/l2d/rq13.log
