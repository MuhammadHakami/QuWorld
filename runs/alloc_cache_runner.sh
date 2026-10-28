#!/bin/bash
# Allocation-strategy study (33-allocation-strategy-study.md): frozen-trunk caches, one at a time, GPU-capped.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=16
S=runs/alloc/status
build() {  # name kind bench extra...
  local tag=$1; shift
  [ -f runs/alloc/cache/$tag/test.pt ] && return
  for attempt in 1 2 3; do
    runs/capped.sh alloc-cache 24 .venv/bin/python paper/alloc/cache.py "$@" >> runs/alloc/cache_$tag.log 2>&1 && break
    echo "CRASH cache $tag attempt $attempt $(date -Is)" >> $S; sleep 60
  done
  echo "CACHE_DONE $tag $(grep -E '^\[cache\] test.pt' runs/alloc/cache_$tag.log | tail -1) $(date -Is)" >> $S
}
build turbulent_radiative_layer_2D well turbulent_radiative_layer_2D
build realpde_unet realpde unet --test 100000
build realpde_fno realpde fno --test 100000
build rayleigh_benard well rayleigh_benard
build shear_flow well shear_flow
build planetswe well planetswe
echo "ALL_CACHES_DONE $(date -Is)" >> $S
