#!/usr/bin/env bash
# Vehicle routing, RouteFinder (Table 1 bottom row, gap % to reference, lower is better): the released RF-Transformer
# fine-tuned with RouteFinder's own efficient adapter layers (EAL) to the unseen multi-depot variants; arms vanilla vs
# route30 (routed 30-atom node register), seeds 1-5, bf16, RouteFinder's defaults (10 epochs x 10,000 instances,
# lr 3e-4). Follows runs/rf_eal_runner.sh with the routed arm of doc 35 §16. The qubit ladder (fp32) is
# runs/rf_full_runner.sh; its figure comes from paper/vrp/rf_ladder.py.
# Needs baseline/routefinder (https://github.com/ai4co/routefinder) and .venv-vrp (rl4co 0.7.0, torchrl 0.14).
#   bash reproduce/routing_routefinder.sh [data|strata|train|all]
source "$(dirname "$0")/common.sh"
VPY=$(pwd)/.venv-vrp/bin/python

if step data; then   # RouteFinder's data and checkpoints from Hugging Face (ai4co/routefinder)
  (cd baseline/routefinder && $VPY scripts/download_hf.py)
fi

if step strata; then # fixed VALIDATION instances (seed 0) for every unseen variant, drawn with RouteFinder's generator
  $VPY paper/strat/routing/make_strata.py
fi

if step train; then
  for seed in 1 2 3 4 5; do for v in vanilla route30; do
    done_if runs/routefinder/eal_md_${v}_s${seed}.json && continue
    say "routefinder EAL $v seed $seed"
    (cd baseline/routefinder && $VPY ../../paper/vrp/run_rf_eal.py $v --setting md --seed $seed)
  done; done
fi
