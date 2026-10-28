#!/bin/bash
# Doc 37: metric-aligned gated fine-tuning on MHD_64 (3-D), after the RB lane-1 ablations free the GPU.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
while ! grep -q "^FT_LANE1_DONE" $S; do sleep 120; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
for v in mr30ryd vanilla; do
  [ -f runs/alloc/ft/MHD_64_${v}_vrmse_gate_s1.json ] && continue; mem_gate
  QW_GPU_GB=24 timeout 9000 runs/capped.sh ftm-$v 40 .venv/bin/python paper/alloc/finetune_gated.py MHD_64 $v --loss vrmse --head gate \
    --budget 3600 --val-every 900 --valk 1 --testk 1 --seed 1 --lr 2e-5 --gate-pretrain 1500 --batch 2 >> runs/alloc/ft.log 2>&1 \
    || echo "CRASH ft MHD_64 $v $(date -Is)" >> $S
done
echo "FT_MHD_DONE $(date -Is)" >> $S
