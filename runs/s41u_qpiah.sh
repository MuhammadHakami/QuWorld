#!/bin/bash
# Fusion §41u on the Spark (merphistophline handed to Learn2Design, 2026-10-11): register heads on the J1AH input set (J1QPIpiAH: PI-KAN +
# simulated 30-atom Rydberg layer, all data + thz + hfull), seeds 1-2, ~4 h each; after the §41w out-of-fold thp re-test.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q S41W_DONE $L/s41w_oof.log 2>/dev/null; do sleep 60; done
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
for s in 1 2; do
  [ -f $L/head_boundary_v8_J1QPIpiAH_s$s.pt ] && continue
  flock runs/strat/gpu.lock env STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 QW_MAGFIX=X_magfix QW_GPU_GB=60 runs/capped.sh s41u 60 .venv/bin/python \
    paper/strat/fusion/hpo_qpikan.py --fixed "$PK" --tag J1QPIpiAH --epochs 60 --seed $s > $L/J1QPIpiAH_s$s.log 2>&1 || echo "CRASH J1QPIpiAH s$s $(date -Is)" >> $L/J1QPIpiAH_s$s.log
done
echo "S41U_DONE $(date -Is)" >> $L/J1QPIpiAH_s1.log
