#!/bin/bash
# merphistophline: §40q global-best donor scan (value-only, 512 donors by saved loss, runs 1 3 4 6 8). Replaces rq11.
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
while pgrep -f "[d]onor_scan.py --topo-seeds" > /dev/null; do sleep 15; done
$PY paper/l2d/donor_scan.py --runs 3 4 1 6 8 --near 512 --rand 0 --by loss > runs/l2d/donor_scan_best.log 2>&1
echo "L2D_RQ12_DONE $(date -Is)" >> runs/l2d/rq12.log
