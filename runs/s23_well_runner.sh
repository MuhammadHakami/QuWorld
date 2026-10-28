#!/bin/bash
# doc 35 §23: Well rows, same two-stage recipe as the published neck models: neck fine-tune (6000 steps, lr 1e-4, batch 4,
# seed 1337) -> gated metric-aligned fine-tune (budget 3600 s, lr 2e-5, gate pre-training 3000) -> full TEST once.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
chain() { local ds=$1 extra=$2
  for v in q30-geo=3d q30-geo=3d-mlp; do
    while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 30; done
    [ -f runs/well2d/$ds/cnext/${v}_s1337.pt ] || QW_GPU_GB=16 timeout 20000 runs/capped.sh s23n-$ds 32 \
      uv run --no-sync python paper/phase_b/finetune_well2d.py $ds $v 1337 14400 1e-4 6000 >> runs/well2d/$ds/cnext/${v}_s1337.log 2>&1 || echo "CRASH s23 neck $ds $v $(date -Is)" >> $S
    QW_GPU_GB=16 timeout 9000 runs/capped.sh s23g-$ds 32 .venv/bin/python paper/alloc/finetune_gated.py $ds $v --loss vrmse --head gate \
      --budget 3600 --val-every 600 --valk 1 --seed 1 --lr 2e-5 --gate-pretrain 3000 $extra >> runs/alloc/ft.log 2>&1 || echo "CRASH s23 gated $ds $v $(date -Is)" >> $S
    QW_GPU_GB=16 timeout 9000 runs/capped.sh s23t-$ds 32 .venv/bin/python paper/alloc/finetune_gated.py $ds $v --loss vrmse --head gate \
      --seed 1 --lr 2e-5 $extra --full-test >> runs/alloc/ft.log 2>&1 || echo "CRASH s23 test $ds $v $(date -Is)" >> $S
  done; echo "S23_WELL_${ds}_DONE $(date -Is)" >> $S; }
chain shear_flow "--testk 2" & chain rayleigh_benard "--tag-suffix _pre" & wait
