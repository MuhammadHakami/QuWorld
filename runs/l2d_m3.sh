#!/bin/bash
# mephiles (L2D only, CUDA): §40n v3 screen, runs 3 and 4 (after the v1 run-8 reference).
cd ~/Documents/paper/quworld; export XLA_PYTHON_CLIENT_PREALLOCATE=false PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "[l]2d_m2.sh" > /dev/null || pgrep -f "[r]un_submission.py" > /dev/null; do sleep 30; done
for r in 3 4; do
  $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v3 --kw '{"k": 512, "lr_steps": [0.01, 0.003], "step_patience": 500}' > runs/l2d/sub_v3_run$r.log 2>&1
done
echo "L2D_M3_DONE $(date -Is)" >> runs/l2d/m3.log
