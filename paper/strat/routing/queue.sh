#!/bin/bash
# §37 routing queue: one GPU job at a time through runs/strat/gpu.lock, 12 GiB cap, 30-min timeout, >= 30 GB free.
# Every launch runs the whole ordered job list in one process (one lock acquisition); finished jobs are skipped, and
# the process stops itself at 26 min, so repeated launches resume where the previous one stopped.
#   bash paper/strat/routing/queue.sh  (run from the repo root; log: runs/strat/routing/queue.log)
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/strat/routing/queue.log
PY=.venv-vrp/bin/python
JOBS="eal0:g8_bf16:128:offval,eal0:g8_fp32:128:offval,q30-geo=3d:g8_bf16:128,q30-geo=3d:g8_fp32:128,q30-geo=3d:g8_bf16:128:offval"
JOBS="$JOBS,vanilla:g8_bf16:128,q30-geo=3d-mlp:g8_bf16:128,q30-geo=3d:s8x4t05:128,q30-geo=3d:g32sym:32"
JOBS="$JOBS,vanilla:g8_fp32:128,q30-geo=3d-mlp:g8_fp32:128,q30-geo=3d:g8_fp16:128,q30-geo=3d:s8x4:128"
JOBS="$JOBS,vanilla:s8x4t05:128,q30-geo=3d-mlp:s8x4t05:128"
remaining() { $PY - "$JOBS" <<'EOF'
import os, sys
n = 0
for spec in sys.argv[1].split(","):
    arm, st, b, *f = spec.split(":")
    n += not os.path.exists(f"runs/strat/routing/scores/{arm}__{st}{'__offval' if f else ''}.json")
print(n)
EOF
}
for round in $(seq 1 8); do
  left=$(remaining); [ "$left" -eq 0 ] && break
  for i in $(seq 1 60); do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 30 ] && break
    echo "$(date +%T) wait: MemAvailable ${avail} GB < 30" >> $LOG; sleep 60
  done
  echo "$(date +%T) ROUND $round: $left jobs left (MemAvailable ${avail} GB)" >> $LOG
  flock runs/strat/gpu.lock bash -c "avail=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo); echo \"\$(date +%T) LOCK acquired (MemAvailable \${avail} GB)\"; [ \"\$avail\" -ge 30 ] || exit 75; timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_vrp 16 $PY paper/strat/routing/eval_md.py --jobs '$JOBS' --budget 1560" >> $LOG 2>&1
  echo "$(date +%T) EXIT $? round $round" >> $LOG
done
echo "$(date +%T) QUEUE_END ($(remaining) left)" >> $LOG
