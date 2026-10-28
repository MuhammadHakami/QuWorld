#!/bin/bash
# Doc 36 §6: gate + arms on TRL2D 3-seed ensemble-mean trunks (register-neck and vanilla), full TEST.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
arms="none cls_only static_lin adrc_ryd adrc_gnn adrc_mlp frozen_ryd adrc_gutz rand_ryd"
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
for v in mr30ryd vanilla; do
  tag=turbulent_radiative_layer_2D:gate_ens3_$v; dir=${tag/:/__}
  ck=runs/phase_b/${v}_s1337.pt,runs/phase_b/${v}_s1338.pt,runs/phase_b/${v}_s1339.pt
  mem_gate; [ -f runs/alloc/adrc_cache/$dir/test.pt ] || ADRC_BANDS=5 ADRC_BLEND=5 ADRC_TEST_ALL=1 ADRC_CKPT=$ck ADRC_VARIANT=$v \
    QW_GPU_GB=24 timeout 14400 runs/capped.sh bgp-ce 40 .venv/bin/python paper/alloc/adrc_cache.py $tag >> runs/alloc/bgp_cache.log 2>&1 || { echo "CRASH bgp cache $tag $(date -Is)" >> $S; continue; }
  for seed in 1 2 3; do for arm in $arms; do
    [ -f runs/alloc/adrc_results/${dir}_${arm}_gate_s${seed}.json ] && continue
    QW_GPU_GB=12 timeout 7200 runs/capped.sh bgp-e 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed --gate >> runs/alloc/adrc.log 2>&1 || echo "CRASH bgp $dir $arm s$seed $(date -Is)" >> $S
  done; done
  echo "BGP_DONE $tag $(date -Is)" >> $S
done
