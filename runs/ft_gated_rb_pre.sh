#!/bin/bash
# Doc 37: RB under the two-stage schedule (exact gate pre-training on 3,000 TRAIN windows, then joint, trunk 2e-5),
# three matched trunks, after the RB v1 factorial (lane 2 of runs/ft_gated_runner.sh) finishes.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
while ! grep -q "^FT_RB_DONE" $S; do sleep 120; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 60; done; }
for v in mr30ryd vanilla mlp30; do
  [ -f runs/alloc/ft/rayleigh_benard_${v}_vrmse_gate_s1_pre.json ] && continue; mem_gate
  QW_GPU_GB=16 timeout 9000 runs/capped.sh ftp-$v 32 .venv/bin/python paper/alloc/finetune_gated.py rayleigh_benard $v --loss vrmse --head gate \
    --budget 3600 --val-every 600 --valk 1 --seed 1 --lr 2e-5 --gate-pretrain 3000 --tag-suffix _pre >> runs/alloc/ft.log 2>&1 || echo "CRASH ft rb pre $v $(date -Is)" >> $S
done
echo "FT_RB_PRE_DONE $(date -Is)" >> $S
