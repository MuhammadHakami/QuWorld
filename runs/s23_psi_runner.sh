#!/bin/bash
# doc 35 §23: fusion equilibrium (Sophelio starter U-Net, fixed recipe: 50 epochs, batch 256, lr 1e-3), 3 seeds.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
for seed in 1337 1338 1339; do for v in q30-geo=3d q30-geo=3d-mlp; do
  [ -f "runs/sophelio/${v}_s${seed}.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 24 ]; do sleep 5; done
  QW_GPU_GB=10 timeout 5400 runs/capped.sh s23psi 20 .venv/bin/python paper/sophelio/train_hybrid.py --variant "$v" --seed $seed >> runs/s23_psi.log 2>&1 &
done; done; wait; echo "S23_PSI_DONE $(date -Is)" >> runs/alloc/status
