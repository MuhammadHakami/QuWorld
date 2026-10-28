#!/bin/bash
# merphistophline (one ROCm process at a time): §40l-b v2b screen (restart from best with fresh Adam at lr 0.01 then 0.003;
# runs 1 and 8, 3,000 evaluations) -> §40j donor scan. Replaces rq9 (v2lr run 1 failed; run 8 stopped).
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
while pgrep -f "[r]un_submission.py|[e]val_cands.py|[v]alidate_submission.py|[p]lateau_probe.py|[f]astsim_check.py" > /dev/null; do sleep 30; done
for r in 1 8; do
  $PY paper/l2d/run_submission.py --run $r --evals 3000 --tag v2b --kw '{"lr_steps": [0.01, 0.003], "step_patience": 500}' \
    > runs/l2d/sub_v2b_run$r.log 2>&1
done
$PY paper/l2d/donor_scan.py --runs 3 4 6 9 1 2 5 7 8 10 --near 768 --rand 256 > runs/l2d/donor_scan.log 2>&1
echo "L2D_RQ10_DONE $(date -Is)" >> runs/l2d/rq10.log
