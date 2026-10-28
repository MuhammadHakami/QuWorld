#!/usr/bin/env bash
# Vehicle routing, RRNCO asymmetric CVRP (Table 1 bottom row, gap %): the released RRNet (rcvrp, epoch 199) as the
# frozen classical part with a routed 30-atom register on its node latent, both arms trained in one process, seeds 1-3,
# 10 epochs x 10,000 instances, lr 1e-4. Follows runs/s41b_runner.sh. "Our best" adds test-time local search
# (paper/vrp/rrnco_ls.py, reported separately with a dagger).
# Needs baseline/rrnco (https://github.com/ai4co/real-routing-nco) and .venv-vrp.
#   bash reproduce/routing_rrnco.sh [data|train|all]
source "$(dirname "$0")/common.sh"
VPY=$(pwd)/.venv-vrp/bin/python

if step data; then   # RRNCO's real-city data and checkpoints from Hugging Face (ai4co/rrnco)
  (cd baseline/rrnco && $VPY scripts/download_hf.py)
fi

if step train; then
  for s in 1 2 3; do
    done_if runs/routefinder/rrnco_hybrid_s${s}_e10_lr0.0001.json && continue
    (cd baseline/rrnco && $VPY ../../paper/vrp/rrnco_hybrid.py --seed $s --epochs 10 --train-size 10000 --lr 1e-4)
  done
fi
