#!/bin/bash
# doc 35 §37b: rayleigh_benard full TEST, x-symmetry averaging arms; one ~45-min slice per GPU-lock acquisition
# (a registered TEST run, not exploration: 45-min slices fit ~3 shards of ~860 s under contention; shards resume).
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
L=runs/strat/rb/test_s37b.log; S=runs/alloc/status
for i in $(seq 1 60); do
  grep -q '^S37B_ALL_SHARDS_DONE' $L 2>/dev/null && break
  flock runs/strat/gpu.lock bash -c 'until [ $(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo) -ge 30 ]; do sleep 60; done
    QW_GPU_GB=12 timeout 2700 runs/capped.sh s37b 16 uv run --no-sync python paper/strat/rb/score_test_tta.py --budget 2550' >> $L 2>&1
  sleep 20
done
uv run --no-sync python paper/strat/rb/score_test_tta.py --summarize >> $L 2>&1
grep -q '^S37B_ALL_SHARDS_DONE' $L && echo "S37B_DONE $(date -Is)" >> $S || echo "S37B INCOMPLETE $(date -Is)" >> $S
