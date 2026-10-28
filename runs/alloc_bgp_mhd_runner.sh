#!/bin/bash
# Band-gated persistence blend + ADR-C v2 (doc 36): classical frequency-aware anchor, band-wise POD components, 8 arms x 3 seeds.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
arms="none cls_only static_lin adrc_ryd adrc_gnn adrc_mlp frozen_ryd adrc_gutz rand_ryd"
one() { # row variant(or -) test_all
  local row=$1 v=$2 all=$3 tag dir ck=""
  if [ "$v" = "-" ]; then tag=$row:gate_released; else tag=$row:gate_${v}_s1337; ck=$([ $row = MHD_64 ] && echo runs/phase_b_mhd/${v}_s1337.pt || echo runs/well2d/$row/cnext/${v}_s1337.pt); fi
  dir=${tag/:/__}
  [ -f runs/alloc/adrc_cache/$dir/test.pt ] || ADRC_BANDS=5 ADRC_BLEND=5 ADRC_TEST_ALL=1 ADRC_CKPT=$ck ADRC_VARIANT=${v/-/vanilla} \
    QW_GPU_GB=24 timeout 14400 runs/capped.sh bgp-cm 40 .venv/bin/python paper/alloc/adrc_cache.py $tag >> runs/alloc/bgp_cache.log 2>&1 || { echo "CRASH bgp cache $tag $(date -Is)" >> $S; return; }
  for seed in 1 2 3; do for arm in $arms; do
    [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 7200 runs/capped.sh bgp-m 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate >> runs/alloc/adrc.log 2>&1 || echo "CRASH bgp $dir $arm s$seed $(date -Is)" >> $S
  done; done
  echo "BGP_DONE $tag $(date -Is)" >> $S
}
one MHD_64 vanilla 1
one MHD_64 mr30ryd 1
