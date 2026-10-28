#!/bin/bash
# Cache wave 3 (replaces runner2's wait on planetswe): Pelzer 1HP and MHD_64.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_GPU_GB=16
S=runs/alloc/status
build() { local tag=$1; shift
  [ -f runs/alloc/cache/$tag/test.pt ] && return
  for attempt in 1 2 3; do
    runs/capped.sh alloc-cache 24 .venv/bin/python paper/alloc/cache.py "$@" >> runs/alloc/cache_$tag.log 2>&1 && break
    echo "CRASH cache $tag attempt $attempt $(date -Is)" >> $S; sleep 60
  done
  echo "CACHE_DONE $tag $(grep -E '^\[cache\] test.pt' runs/alloc/cache_$tag.log | tail -1) $(date -Is)" >> $S
}
build pelzer pelzer runs/pelzer_unet/pretrain_lr0.001_s1337.pt --batch 50 --levels 4
build MHD_64 well3d MHD_64:data/cnext_mhd_ckpt --batch 2 --levels 4
echo "ALL_CACHES_DONE $(date -Is)" >> $S
