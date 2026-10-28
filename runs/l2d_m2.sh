#!/bin/bash
# mephiles (L2D only, CUDA - reliable): after the diagnostic, the §40l-b v2b screen on CUDA (runs 1 and 8, 3,000
# evaluations) - the ROCm v2b run 1 froze with non-finite evaluations (Steam/Proton shares that GPU).
cd ~/Documents/paper/quworld; export XLA_PYTHON_CLIENT_PREALLOCATE=false PY=baseline/Learn2Design-2026/.venv/bin/python
while pgrep -f "[l]2d_m_diag.sh" > /dev/null || pgrep -f "[r]un_submission.py" > /dev/null; do sleep 30; done
for r in 1 8; do
  $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v2b_cuda --kw '{"lr_steps": [0.01, 0.003], "step_patience": 500}' > runs/l2d/sub_v2b_cuda_run$r.log 2>&1
done
$PY paper/l2d/run_submission.py --run 8 --evals 3000 --tag v1_cuda > runs/l2d/sub_v1_cuda_run8.log 2>&1
echo "L2D_M2_DONE $(date -Is)" >> runs/l2d/m2.log
