#!/usr/bin/env bash
# Geothermal heat plume (Pelzer 1HP, Table 1 neck row): pretrain 1HP_NN's own U-Net per seed (learning rate picked on
# VALID at seed 1337), then tune every arm with Optuna (12 trials, objective = mean VALID over seeds 1337 + 1338) and
# score once on TEST with seeds 1337-1339. Follows runs/pelzer_unet_runner.sh (pretraining) and runs/hpo_ms_runner.sh.
# Needs baseline/1HP_NN (https://github.com/JuliaPelzer/1HP_NN).
# The Optuna benchmark (paper/hpo/bench_pelzer.py) loads the lr 0.001 trunks, the VALID-best rate in the paper.
#   bash reproduce/heat_plume.sh [data|pretrain|train|all]
source "$(dirname "$0")/common.sh"
O=runs/pelzer_unet; mkdir -p $O

if step data; then   # DaRUS doi:10.18419/darus-3653, 402 files, md5-checked, dataverse folders kept
  py paper/geothermal/fetch_darus.py --doi 10.18419/darus-3653 --out data/geothermal/darus-3653
fi

if step pretrain; then  # 3,000 steps; learning rate from {1e-3, 3e-4, 1e-4} by VALID MAE at seed 1337
  for lr in 0.001 0.0003 0.0001; do                                    # file names use the %g form
    done_if $O/pretrain_lr${lr}_s1337.json || py paper/geothermal/train_pelzer_unet.py pretrain $lr 1337 3000
  done
  best=$(py -c "import glob, json; print(min((json.load(open(f))['best_val_mae_c'], f) for f in glob.glob('$O/pretrain_lr*_s1337.json'))[1].split('_lr')[1].split('_s')[0])")
  say "best pretraining learning rate: $best"
  for seed in 1338 1339; do
    done_if $O/pretrain_lr${best}_s${seed}.json || py paper/geothermal/train_pelzer_unet.py pretrain $best $seed 3000
  done
fi

if step train; then  # arms: none (classical) | mlp (matched MLP) | q (30-atom register) at the deepest encoder level
  for arm in none mlp q; do
    done_if runs/hpo/pelzer/${arm}_ms2_final.json && continue
    py paper/hpo/driver.py pelzer $arm search --trials 12 --search-seeds 1337 1338 --study ms2
    py paper/hpo/driver.py pelzer $arm final --study ms2
  done
fi
