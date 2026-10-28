#!/bin/bash
# Sub-study B (doc 33): analog vs digital inductive-bias probe. Usage: runs/alloc_probe_runner.sh <seed...>
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=4
for seed in "$@"; do
  for core in mlp mlp+ mlpX4 ryd gutz; do
    for w in 0.5 1 2 4 8; do timeout 3600 runs/capped.sh probe 6 .venv/bin/python paper/alloc/bias_probe.py --target analog --omega $w --core $core --seed $seed >> runs/alloc/probe.log 2>&1; done
    for t in digital parity; do timeout 3600 runs/capped.sh probe 6 .venv/bin/python paper/alloc/bias_probe.py --target $t --core $core --seed $seed >> runs/alloc/probe.log 2>&1; done
  done
done
echo "PROBE_DONE seeds $* $(date -Is)" >> runs/alloc/status
