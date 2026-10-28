#!/bin/bash
# Doc 36 §7: fast iteration on stratified samples (trajectory x 8 time bins; VALID k=2, TEST k=4), full TEST only for
# the VALID-selected stack. Also finishes the remaining full-TEST shear/TRL2D arms of the stopped runners.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
arms="none cls_only static_lin adrc_ryd adrc_gnn adrc_mlp frozen_ryd adrc_gutz rand_ryd"
arms_of() { local dir=$1 tag=$2; shift 2
  for seed in 1 2 3; do for arm in $arms; do
    [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 7200 runs/capped.sh gst 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate >> runs/alloc/adrc.log 2>&1 || echo "CRASH gst $dir $arm s$seed $(date -Is)" >> $S
  done; done; echo "BGP_DONE $tag $(date -Is)" >> $S; }
cache() { # tag ckpt variant env...
  local tag=$1 ck=$2 v=$3; shift 3; local dir=${tag/:/__}
  [ -f runs/alloc/adrc_cache/$dir/test.pt ] || env ADRC_BANDS=5 ADRC_BLEND=5 ADRC_CKPT=$ck ADRC_VARIANT=$v "$@" \
    QW_GPU_GB=24 timeout 14400 runs/capped.sh gst-c 40 .venv/bin/python paper/alloc/adrc_cache.py $tag >> runs/alloc/bgp_cache.log 2>&1 || { echo "CRASH gst cache $tag $(date -Is)" >> $S; return 1; }
}
for v in mr30ryd vanilla; do
  tag=rayleigh_benard:gatest_${v}_s1337
  cache $tag runs/well2d/rayleigh_benard/cnext/${v}_s1337.pt $v ADRC_TEST_STRAT=4 ADRC_VALID_STRAT=2 && arms_of ${tag/:/__} $tag
done
arms_of shear_flow__gate_mr30ryd_s1337 shear_flow:gate_mr30ryd_s1337
arms_of turbulent_radiative_layer_2D__gate_released turbulent_radiative_layer_2D:gate_released
tag=shear_flow:gate_vanilla_s1337; cache $tag runs/well2d/shear_flow/cnext/vanilla_s1337.pt vanilla ADRC_TEST_ALL=1 && arms_of ${tag/:/__} $tag
