#!/bin/bash
# merphistophline (one ROCm process at a time): after the running §40i probe: §40k fastsim ROCm check -> §40l v2 screen
# (lr decay; runs 1 and 8, 3,000 evaluations) -> §40j donor scan. Replaces rq8.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
while pgrep -f "[r]un_submission.py|[e]val_cands.py|[v]alidate_submission.py|[p]lateau_probe.py" > /dev/null; do sleep 30; done
$PY paper/l2d/fastsim_check.py --run 1 --donors 8 --rand 4 --modes c64,mixed > runs/l2d/fastsim_check_rocm.log 2>&1
for r in 1 8; do
  $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v2lr --kw '{"lr_decay": 0.3, "lr_patience": 300, "lr_min": 0.003}' \
    > runs/l2d/sub_v2lr_run$r.log 2>&1
done
$PY paper/l2d/donor_scan.py --runs 3 4 6 9 1 2 5 7 8 10 --near 768 --rand 256 > runs/l2d/donor_scan.log 2>&1
echo "L2D_RQ9_DONE $(date -Is)" >> runs/l2d/rq9.log
