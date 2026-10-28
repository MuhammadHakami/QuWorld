#!/usr/bin/env bash
# Convection (The Well rayleigh_benard): Table 1 routed row and the pooled-neck row. Follows runs/well2d_runner.sh
# (neck arms, released-checkpoint anchor) and runs/s32_runner.sh (routed row, full TEST: Table 1 reads
# runs/alloc/adrc_results_s32).
#   bash reproduce/convection.sh [data|anchor|train|routed|all]
source "$(dirname "$0")/common.sh"
DS=rayleigh_benard; O=runs/well2d/$DS/cnext
export BACKBONE=cnext BATCH=4 VAL_PER_TRAJ=8 VAL_EVERY_S=300 MEM_FRACTION=0.3

if step data; then
  hf polymathic-ai/$DS --repo-type dataset --local-dir data/well/$DS                       # 370 GB
  hf polymathic-ai/UNetConvNext-$DS --local-dir data/cnext_rb_ckpt
  py paper/phase_b/well2d.py $DS valid                                                      # loader == WellDataset check
  py paper/strat/rb/make_strata.py                                                          # fixed stratified VALID sample
fi

if step anchor; then
  mkdir -p $O; done_if $O/official_test.csv || BATCH=16 py paper/phase_b/eval_well2d.py $DS official vanilla $O/official_test.csv
fi

if step train; then   # neck insertion: 6,000 steps, batch 4, lr 1e-4, best-on-VALID checkpoint, one TEST eval each
  for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
    done_if $O/${v}_s${seed}.pt || py paper/phase_b/finetune_well2d.py $DS $v $seed 14400 1e-4 6000
    done_if $O/${v}_s${seed}_test.csv || BATCH=16 py paper/phase_b/eval_well2d.py $DS $O/${v}_s${seed}.pt $v $O/${v}_s${seed}_test.csv
  done; done
fi

if step routed; then  # stratified VALID for selection, every TEST window for the table
  routed_cache $DS:gatestfull_vanilla_s1337 $O/vanilla_s1337.pt ADRC_VALID_STRAT=2 ADRC_TEST_ALL=1
  routed_arms ${DS}__gatestfull_vanilla_s1337 runs/alloc/adrc_results_s32
fi
