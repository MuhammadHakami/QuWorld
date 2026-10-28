#!/bin/bash
# Doc 36 §6-8: remaining gate caches (matched vanilla trunks, TRL2D ensembles), each followed by its causal-gate arms.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
cache() { local tag=$1 ck=$2 v=$3; shift 3; local dir=${tag/:/__}
  [ -f runs/alloc/adrc_cache/$dir/test.pt ] && return 0; mem_gate
  env ADRC_BANDS=5 ADRC_BLEND=5 ADRC_CKPT=$ck ADRC_VARIANT=$v "$@" QW_GPU_GB=24 timeout 14400 runs/capped.sh g2c 40 \
    .venv/bin/python paper/alloc/adrc_cache.py $tag >> runs/alloc/bgp_cache.log 2>&1 || { echo "CRASH g2 cache $tag $(date -Is)" >> $S; return 1; }; }
cache rayleigh_benard:gatest_vanilla_s1337 runs/well2d/rayleigh_benard/cnext/vanilla_s1337.pt vanilla ADRC_TEST_STRAT=4 ADRC_VALID_STRAT=2 \
  && runs/alloc_gate2_runner.sh D1 rayleigh_benard__gatest_vanilla_s1337 &
p=runs/phase_b
cache turbulent_radiative_layer_2D:gate_ens3_mr30ryd $p/mr30ryd_s1337.pt,$p/mr30ryd_s1338.pt,$p/mr30ryd_s1339.pt mr30ryd ADRC_TEST_ALL=1
cache turbulent_radiative_layer_2D:gate_ens3_vanilla $p/vanilla_s1337.pt,$p/vanilla_s1338.pt,$p/vanilla_s1339.pt vanilla ADRC_TEST_ALL=1
runs/alloc_gate2_runner.sh D2 turbulent_radiative_layer_2D__gate_ens3_mr30ryd turbulent_radiative_layer_2D__gate_ens3_vanilla &
cache shear_flow:gate_vanilla_s1337 runs/well2d/shear_flow/cnext/vanilla_s1337.pt vanilla ADRC_TEST_ALL=1 && runs/alloc_gate2_runner.sh D3 shear_flow__gate_vanilla_s1337
while kill -0 2103439 2>/dev/null; do sleep 60; done
runs/alloc_gate2_runner.sh D4 MHD_64__gate_mr30ryd_s1337
wait
