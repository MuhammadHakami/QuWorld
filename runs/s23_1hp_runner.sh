#!/bin/bash
# doc 35 §23: heat plume (1HP) neck arms, fixed recipe (1500 steps, lr 1e-4, pretrain lr0.001 s1337), 3 seeds.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
CK=runs/pelzer_unet/pretrain_lr0.001_s1337.pt
for seed in 1337 1338 1339; do for v in q30-geo=3d q30-geo=3d-mlp vanilla; do
  [ -f "runs/pelzer_unet/${v}_s${seed}.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ] || [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 5; done
  QW_GPU_GB=8 timeout 3600 runs/capped.sh s23hp 12 .venv/bin/python paper/geothermal/train_pelzer_unet.py arm "$v" $seed 1500 0.0001 $CK >> runs/s23_1hp.log 2>&1 &
done; done; wait; echo "S23_1HP_DONE $(date -Is)" >> runs/alloc/status
