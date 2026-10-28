#!/bin/bash
# Sub-study B2 (doc 33, exploratory): rydlin / mlpsin at omega 4, 8, 16 (+ registered cores at omega 16). Usage: <seed>
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=4; s=$1
for core in mlpsin rydlin; do for w in 4 8 16; do
  timeout 3600 runs/capped.sh probe2 6 .venv/bin/python paper/alloc/bias_probe.py --target analog --omega $w --core $core --seed $s >> runs/alloc/probe2.log 2>&1
done; done
for core in mlp mlp+ mlpX4 ryd; do
  timeout 3600 runs/capped.sh probe2 6 .venv/bin/python paper/alloc/bias_probe.py --target analog --omega 16 --core $core --seed $s >> runs/alloc/probe2.log 2>&1
done
echo "PROBE2_DONE seed $s $(date -Is)" >> runs/alloc/status
