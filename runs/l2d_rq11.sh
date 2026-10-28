#!/bin/bash
# merphistophline (value-only, non-finite counts saved): §40j ranker data - donor scans on 20 fresh topologies (seeds
# 1001-1020; 512 Hamming-nearest + 128 random donors each) for the hybrid ranker's training set (LOTO stays on the 10 retired).
cd ~/Documents/paper/quworld; export JAX_PLATFORMS=rocm; PY=baseline/Learn2Design-2026/.venv-rocm/bin/python
$PY paper/l2d/donor_scan.py --topo-seeds $(seq 1001 1020) --near 512 --rand 128 > runs/l2d/donor_scan_fresh.log 2>&1
echo "L2D_RQ11_DONE $(date -Is)" >> runs/l2d/rq11.log
