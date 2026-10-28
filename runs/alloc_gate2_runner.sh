#!/bin/bash
# Doc 36 §8: gate arms rerun with CAUSAL gate features (the first gate runs normalized by var(y); archived). Seed-major
# so every arm/row gets seed 1 first. Usage: runs/alloc_gate2_runner.sh <name> <cache_dir...> (caches must exist).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; name=$1; shift
arms="none static_lin adrc_ryd adrc_gnn adrc_mlp"          # trimmed Sep 25 00:30 (3 cores + gate + linear ref)
rsarms="adrc_ryd adrc_gnn adrc_mlp"
run1() { local dir=$1 arm=$2 seed=$3 rs=$4 sfx=_gate; [ -n "$rs" ] && sfx=_gate_rs
  [ -f runs/alloc/adrc_results/${dir}_${arm}${sfx}_s${seed}.json ] && return
  QW_GPU_GB=12 timeout 7200 runs/capped.sh g2$name 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate $rs >> runs/alloc/adrc.log 2>&1 || echo "CRASH g2 $dir $arm$sfx s$seed $(date -Is)" >> $S; }
for seed in 1 2 3; do for dir in "$@"; do
  for arm in $arms; do run1 $dir $arm $seed; done
  for arm in $rsarms; do run1 $dir $arm $seed --route_sup; done
done; echo "G2_SEED_DONE $name s$seed $(date -Is)" >> $S; done
echo "G2_DONE $name $(date -Is)" >> $S
