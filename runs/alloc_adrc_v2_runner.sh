#!/bin/bash
# ADR-C v2 registered runs (doc 35 §8): band-wise POD caches (<row>:fb4), then 8 arms x 3 seeds; TRL2D v1 for comparison.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
arms="cls_only static_lin adrc_ryd adrc_gnn adrc_mlp frozen_ryd adrc_gutz rand_ryd"
one() { # tag bands
  local tag=$1 nb=$2 dir=${1/:/__}
  [ -f runs/alloc/adrc_cache/$dir/test.pt ] || ADRC_BANDS=$nb QW_GPU_GB=24 timeout 7200 runs/capped.sh adrc2-c 32 .venv/bin/python paper/alloc/adrc_cache.py $tag >> runs/alloc/adrc_cache.log 2>&1 || { echo "CRASH adrc2 cache $tag $(date -Is)" >> $S; return; }
  for seed in 1 2 3; do for arm in $arms; do
    [ -f runs/alloc/adrc_results/${dir}_${arm}_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 5400 runs/capped.sh adrc2 16 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed >> runs/alloc/adrc.log 2>&1 || echo "CRASH adrc2 $dir $arm s$seed $(date -Is)" >> $S
  done; done
  echo "ADRC2_DONE $tag $(date -Is)" >> $S
}
for r in pelzer rayleigh_benard shear_flow turbulent_radiative_layer_2D; do one $r:fb4 4; done
one turbulent_radiative_layer_2D 0
one MHD_64:fb4 4
