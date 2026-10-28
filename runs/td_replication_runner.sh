#!/bin/bash
# Doc 38: 5-seed replication of the TRL2D TD fine-tune (one lane per arm; seeds 1340 then 1341), then the round-2 evals.
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True THROTTLE_MS=15
v=$1; S=runs/phase_b2/replication_status; PY=.venv/bin/python
for seed in 1340 1341; do
  log=runs/phase_b2/td_${v}_s${seed}.log
  for attempt in $(seq 1 12); do
    grep -q "target_reached" "$log" 2>/dev/null && break
    while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 60; done
    echo "arm $v s$seed attempt $attempt $(date -Is)" >> $S
    runs/capped.sh td-$v-$seed 40 $PY paper/phase_b/finetune_td.py $v $seed 2400 1e-4 10000 4 >> $log 2>&1
  done
  grep -q "target_reached" "$log" || { echo "CRASH td $v s$seed INCOMPLETE $(date -Is)" >> $S; continue; }
  ck=runs/phase_b2/${v}_s${seed}.pt; n=${v}_s${seed}
  $PY paper/phase_b/eval_official.py $ck runs/phase_b2/test_$n.csv > runs/phase_b2/test_$n.log 2>&1 || echo "CRASH eval_official $n $(date -Is)" >> $S
  $PY paper/phase_b/eval_rollout_pertraj.py $ck $n > runs/phase_b2/rollout_$n.log 2>&1 || echo "CRASH pertraj $n $(date -Is)" >> $S
  $PY paper/phase_b/eval_rollout_multistart.py $ck $n > runs/phase_b2/multistart_$n.log 2>&1 || echo "CRASH multistart $n $(date -Is)" >> $S
  echo "REPL_DONE $n $(date -Is)" >> $S
done
echo "REPL_LANE_DONE $v $(date -Is)" >> $S
