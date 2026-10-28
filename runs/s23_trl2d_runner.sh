#!/bin/bash
# doc 35 §23: TRL2D, same two-stage recipe as the published models: round-1 neck fine-tune (finetune.py, 5000 steps,
# lr 1e-4, budget 2400 s, seed 1337) -> gated fine-tune (budget 3600 s, lr 2e-5, gate pre-training 3000, testk 2) -> full TEST.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
until grep -q S23_WELL_shear_flow_DONE $S && grep -q S23_WELL_rayleigh_benard_DONE $S; do sleep 60; done
for v in q30-geo=3d q30-geo=3d-mlp; do
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 40 ]; do sleep 60; done
  [ -f runs/phase_b/${v}_s1337.pt ] || QW_GPU_GB=24 timeout 20000 runs/capped.sh s23trl 48 .venv/bin/python paper/phase_b/finetune.py $v 1337 2400 1e-4 5000 \
     >> runs/s23_trl2d.log 2>&1 || echo "CRASH s23 trl2d neck $v $(date -Is)" >> $S
  QW_GPU_GB=16 timeout 9000 runs/capped.sh s23trlg 32 .venv/bin/python paper/alloc/finetune_gated.py turbulent_radiative_layer_2D $v --loss vrmse \
     --head gate --budget 3600 --val-every 600 --valk 1 --testk 2 --seed 1 --lr 2e-5 --gate-pretrain 3000 >> runs/alloc/ft.log 2>&1 || echo "CRASH s23 trl2d gated $v $(date -Is)" >> $S
  QW_GPU_GB=16 timeout 9000 runs/capped.sh s23trlt 32 .venv/bin/python paper/alloc/finetune_gated.py turbulent_radiative_layer_2D $v --loss vrmse \
     --head gate --seed 1 --lr 2e-5 --testk 2 --full-test >> runs/alloc/ft.log 2>&1 || echo "CRASH s23 trl2d test $v $(date -Is)" >> $S
done; echo "S23_TRL2D_DONE $(date -Is)" >> $S
