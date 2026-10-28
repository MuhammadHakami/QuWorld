#!/bin/bash
# Doc 35 §21 amendment (10:43): GNN competitors with the teacher's pair interactions.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
for c in gnn gnnX4; do for s in 1 2 3; do QW_GPU_GB=8 timeout 3000 runs/capped.sh pcg 12 .venv/bin/python paper/alloc/bias_probe.py --target quantum --core $c --seed $s >> runs/alloc/positive_control.log 2>&1 || echo "CRASH pc $c s$s $(date -Is)" >> $S; done; done
echo "PC_GNN_DONE $(date -Is)" >> $S
