#!/usr/bin/env bash
# MHD_64 (The Well, 3-D): Table 1 routed row and the pooled-neck row. Follows paper/phase_b/run_mhd_campaign.sh (neck
# campaign, released-checkpoint anchor) and runs/alloc_bgp_mhd_runner.sh (routed cache).
# "Our best" (Flowers + gated CNextU-net blend) comes from paper/phase_b/train_flowers_mhd.py, finetune_flowers_vrmse.py
# and score_flowers_test.py (see their headers); it needs baseline/flowers.
#   bash reproduce/mhd.sh [data|anchor|train|routed|all]
source "$(dirname "$0")/common.sh"
DS=MHD_64

if step data; then
  hf polymathic-ai/$DS --repo-type dataset --local-dir data/mhd64                          # 159 GB
  mkdir -p data/well && ln -sfn ../mhd64 data/well/$DS                                      # loaders read data/well/MHD_64
  hf polymathic-ai/UNetConvNext-$DS --local-dir data/cnext_mhd_ckpt
  py -c "import sys; sys.path.insert(0, 'paper/phase_b'); import data_ram_mhd; data_ram_mhd.verify()"   # loader == WellDataset
fi

if step anchor; then
  done_if runs/eval_official_cnext_mhd.csv || \
    WELL_DATASET=$DS EVAL_BATCH=8 py paper/phase_b/eval_official.py data/cnext_mhd_ckpt runs/eval_official_cnext_mhd.csv
fi

if step train; then   # neck insertion: 2,400 s budget, lr 1e-4, 20,000 target steps (resumable), one TEST eval each
  for seed in 1337 1338 1339; do for v in vanilla mr30ryd mlp30; do
    ck=runs/phase_b_mhd/${v}_s${seed}.pt; csv=runs/phase_b_mhd/test_${v}_s${seed}.csv
    done_if $ck || py paper/phase_b/finetune_mhd.py $v $seed 2400 1e-4 20000
    done_if $csv || WELL_DATASET=$DS EVAL_BATCH=8 py paper/phase_b/eval_official.py $ck $csv
    py paper/phase_b/paired_stats.py $csv runs/eval_official_cnext_mhd.csv
  done; done
fi

if step routed; then
  routed_cache $DS:gate_vanilla_s1337 runs/phase_b_mhd/vanilla_s1337.pt ADRC_TEST_ALL=1
  routed_arms ${DS}__gate_vanilla_s1337
fi
