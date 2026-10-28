#!/bin/bash
# merphistophline (one ROCm process at a time): after rq4 (v1b run 6): §40k fastsim accuracy/speed on ROCm (after the running §40i plateau
# probe, started by rq5) -> §40j donor scan (all 10 retired topologies). Replaces rq5/rq6.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
while pgrep -f "[l]2d_rq4.sh|[r]un_submission.py|[e]val_cands.py|[v]alidate_submission.py|[p]lateau_probe.py" > /dev/null; do sleep 30; done
$PY paper/l2d/fastsim_check.py --run 1 --donors 8 --rand 4 --modes c64,mixed > runs/l2d/fastsim_check_rocm.log 2>&1
$PY paper/l2d/donor_scan.py --runs 3 4 6 9 1 2 5 7 8 10 --near 768 --rand 256 > runs/l2d/donor_scan.log 2>&1
echo "L2D_RQ7_DONE $(date -Is)" >> runs/l2d/rq7.log
