#!/bin/bash
# Doc 35 §21 positive control (registered 10:37 Sep 26): quantum-generated targets, 5 cores x 3 seeds, probe protocol.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
run() { QW_GPU_GB=8 timeout 3000 runs/capped.sh pc 12 .venv/bin/python paper/alloc/bias_probe.py --target quantum --core $1 --seed $2 >> runs/alloc/positive_control.log 2>&1 || echo "CRASH pc $1 s$2 $(date -Is)" >> $S; }
for s in 1 2 3; do run ryd $s & done
for c in gutz mlp mlp+ mlpX4; do for s in 1 2 3; do run $c $s; done; done
wait; echo "PC_DONE $(date -Is)" >> $S
