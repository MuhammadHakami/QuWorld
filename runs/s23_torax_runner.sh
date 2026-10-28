#!/bin/bash
# doc 35 §23: TORAX-Bench, fixed recipe (1000 steps, k=4 pushforward), 3 seeds; checkpoints + figure dumps kept.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; O=runs/torax_s23; mkdir -p $O
for seed in 1337 1338 1339; do for v in none q30-geo=3d q30-geo=3d-mlp; do
  [ -f "$O/${v}_k4_s${seed}.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 5; done
  QW_GPU_GB=6 timeout 3600 runs/capped.sh s23tx 8 .venv/bin/python paper/torax_bench/train_torax.py "$v" $seed 1000 4 $O >> runs/s23_torax.log 2>&1 &
done; done; wait; echo "S23_TORAX_DONE $(date -Is)" >> runs/alloc/status
