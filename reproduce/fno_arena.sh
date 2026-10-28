#!/usr/bin/env bash
# FNO arena on TRL2D data (Table 1, "FNO arena"): fixed 2,000 steps x {none, mlp30, mr30ryd, cg30} x k in {1, 4}
# (k = pushforward depth) x seeds 1337-1339. Follows paper/run_ag_wave15.sh; the table row is the k = 4 column.
# paper/data.py builds the 64x192 TRL2D cache (data/cache.pt) on first use.
#   bash reproduce/fno_arena.sh [data|train|all]
source "$(dirname "$0")/common.sh"

if step data; then
  hf polymathic-ai/turbulent_radiative_layer_2D --repo-type dataset --local-dir data/trl2d
  mkdir -p data/well && ln -sfn ../trl2d data/well/turbulent_radiative_layer_2D
  py -c "import sys; sys.path.insert(0, 'paper'); import data; data.load()"                 # -> data/cache.pt
fi

if step train; then
  for seed in 1337 1338 1339; do for k in 1 4; do for v in none mlp30 mr30ryd cg30; do
    log=runs/ag15_${v}_k${k}_s${seed}.log
    grep -q val_vrmse_1 $log 2>/dev/null && continue
    say "arena $v k=$k seed $seed"; py paper/train.py $v $seed 900 2000 $k > $log 2>&1
  done; done; done
  grep -h "val_vrmse" runs/ag15_*_k4_s*.log | tail -12
fi
