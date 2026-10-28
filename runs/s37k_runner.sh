#!/bin/bash
# doc 35 §37k: final-Flowers stratified TRAIN (id+comb) and VALID (30 passes) passes, then the CPU blend analysis.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export STRAT_FLOWER_CKPT=runs/phase_b_mhd/flower_vft_s1337.pt STRAT_SHARD_DIR=shards_vft; S=runs/alloc/status
for split in train valid; do
  passes=$([ $split = train ] && echo trainlite || echo full); L=runs/strat/mhd/vft_$split.log
  for i in $(seq 1 8); do
    grep -qE 'windows in [0-9]+s, left 0$' $L 2>/dev/null && break
    flock runs/strat/gpu.lock bash -c "until [ \$(awk '/MemAvailable/ {printf \"%d\", \$2/1048576}' /proc/meminfo) -ge 30 ]; do sleep 60; done
      QW_GPU_GB=16 timeout 3300 runs/capped.sh s37h 24 uv run --no-sync python paper/strat/mhd/eval_strat.py --split $split --passes $passes --budget 3000" >> $L 2>&1
    sleep 5
  done
done
uv run --no-sync python paper/strat/mhd/analyze.py > runs/strat/mhd/analyze_vft.log 2>&1 && echo "S37K_ANALYZED $(date -Is)" >> $S || echo "CRASH s37k analyze $(date -Is)" >> $S
export STRAT_BLEND_OUT=blend_vft.json STRAT_BLEND_TEST_OUT=blend_vft_test.json STRAT_CHECK_DFC=0.11717
uv run --no-sync python paper/strat/mhd/blend_fit.py > runs/strat/mhd/blend_vft_fit.log 2>&1 || { echo "CRASH s37k fit $(date -Is)" >> $S; exit 1; }
flock runs/strat/gpu.lock env QW_GPU_GB=16 timeout 7200 runs/capped.sh s37k 24 uv run --no-sync python paper/strat/mhd/blend_test.py > runs/strat/mhd/blend_vft_test.log 2>&1 \
  && echo "S37K_TEST_DONE $(date -Is)" >> $S || echo "CRASH s37k test $(date -Is)" >> $S
