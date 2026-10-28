#!/bin/bash
# doc 35 §38z: realise the heads in psi - warp chain passes / caps on the VALID sample (X8 otherwise), then L2D v5 runs 1, 4.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
H=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//')
run() { flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag $1 --psi psiF --qb qh50 --heads "$H" --kw "$2" >> $L/s38z.log 2>&1 || echo "CRASH $1 $(date -Is)" >> $L/s38z.log; }
run X8p2 '{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.10},"_passes":2}'
run X8c2 '{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.20},"bw":{"cap":0.10},"kw":{"cap":0.20}}'
run X8p2c2 '{"_steps":["shift","bw","kw","xp","li"],"xp":{"sigma":0.15,"cap":0.20},"bw":{"cap":0.10},"kw":{"cap":0.20},"_passes":2}'
echo "S38Z_DONE $(date -Is)" >> $L/s38z.log
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
exec 9>runs/strat/gpu.lock; flock 9
for r in 1 4; do QW_GPU_GB=30 runs/capped.sh l2d 30 baseline/Learn2Design-2026/.venv/bin/python paper/l2d/run_submission.py --run $r --evals 3000 --tag v5 \
  --kw '{"hop_mode": "xcell", "patience": 500, "seg": 150, "lr_hop": 0.01, "pool_every": 30}' > runs/l2d/sub_v5x_run$r.log 2>&1; done
