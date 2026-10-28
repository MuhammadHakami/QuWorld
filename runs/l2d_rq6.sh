#!/bin/bash
# merphistophline (one ROCm process at a time): §40j donor scan on all 10 retired topologies (hard ones first), after rq5.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm
while pgrep -f "[l]2d_rq4.sh|[l]2d_rq5.sh|[r]un_submission.py|[e]val_cands.py|[v]alidate_submission.py|[p]lateau_probe.py" > /dev/null; do sleep 30; done
baseline/Learn2Design-2026/.venv-rocm/bin/python paper/l2d/donor_scan.py --runs 3 4 6 9 1 2 5 7 8 10 --near 768 --rand 256 \
  > runs/l2d/donor_scan.log 2>&1
