#!/bin/bash
# doc 35 §37h: final-Flowers stratified TRAIN (id+comb) and VALID (30 passes) passes, then the CPU blend analysis.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export STRAT_FLOWER_CKPT=runs/phase_b_mhd/flower_s1337.pt STRAT_SHARD_DIR=shards_final; S=runs/alloc/status
for split in train valid; do
  passes=$([ $split = train ] && echo trainlite || echo full); L=runs/strat/mhd/final_$split.log
  for i in $(seq 1 8); do
    grep -qE 'windows in [0-9]+s, left 0$' $L 2>/dev/null && break
    flock runs/strat/gpu.lock bash -c "until [ \$(awk '/MemAvailable/ {printf \"%d\", \$2/1048576}' /proc/meminfo) -ge 30 ]; do sleep 60; done
      QW_GPU_GB=16 timeout 3300 runs/capped.sh s37h 24 uv run --no-sync python paper/strat/mhd/eval_strat.py --split $split --passes $passes --budget 3000" >> $L 2>&1
    sleep 5
  done
done
uv run --no-sync python paper/strat/mhd/analyze.py > runs/strat/mhd/analyze_final.log 2>&1 && echo "S37H_ANALYZED $(date -Is)" >> $S || echo "CRASH s37h analyze $(date -Is)" >> $S
