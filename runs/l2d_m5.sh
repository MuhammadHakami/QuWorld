#!/bin/bash
# mephiles (CUDA): §40r v4 (v3b + log-length) runs 1 and 3, after the §40o long run.
cd ~/Documents/paper/quworld; export XLA_PYTHON_CLIENT_PREALLOCATE=false PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "[l]2d_m4.sh" > /dev/null || pgrep -f "[r]un_submission.py" > /dev/null; do sleep 30; done
for r in 1 3; do $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v4 --kw '{"loglen": true}' > runs/l2d/sub_v4_run$r.log 2>&1; done
echo "L2D_M5_DONE $(date -Is)" >> runs/l2d/m5.log
