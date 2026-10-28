#!/bin/bash
# Doc 37: end-to-end VRMSE + band-gate fine-tuning on shear_flow and TRL2D (3 matched trunks each), memory-gated lane.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
ft() { local ds=$1 v=$2
  [ -f runs/alloc/ft/${ds}_${v}_vrmse_gate_s1.json ] && return; mem_gate
  QW_GPU_GB=16 timeout 7200 runs/capped.sh ft2-$v 32 .venv/bin/python paper/alloc/finetune_gated.py $ds $v --loss vrmse --head gate \
    --budget 3600 --val-every 600 --valk 1 --testk 2 --seed 1 --lr 2e-5 --gate-pretrain 3000 >> runs/alloc/ft.log 2>&1 || echo "CRASH ft $ds $v $(date -Is)" >> $S; }
for v in mr30ryd vanilla mlp30; do ft shear_flow $v; done
for v in mr30ryd vanilla mlp30; do ft turbulent_radiative_layer_2D $v; done
echo "FT2_DONE $(date -Is)" >> $S
