#!/bin/bash
# Sliding-start rollout evaluations (19-multistart-power-plan.md) over the existing
# round-2 checkpoints, the anchor and the noent arms. Waits for the Gutzwiller arena
# pilot to finish (its 1500 s budget must not be shared), then timeshares with the
# first TD arm (minutes per model). Idempotent: skips models with an npz.
# Launch: setsid nohup bash /home/mohhakami/Documents/paper/quworld/paper/phase_b/run_multistart.sh \
#   >> /home/mohhakami/Documents/paper/quworld/runs/phase_b2/multistart_runner.log 2>&1 < /dev/null &
export PATH="$HOME/.local/bin:$PATH"
set -u
cd /home/mohhakami/Documents/paper/quworld
exec 9> runs/phase_b2/multistart.lock
flock -n 9 || { echo "another run_multistart.sh holds the lock; exiting"; exit 0; }
export PYTHONUNBUFFERED=1
STATUS=runs/phase_b2/multistart_status
echo "multistart start $(date -Is)" >> "$STATUS"
until grep -q "PILOT_DONE" runs/phase_b2/gutz_status 2>/dev/null; do sleep 60; done
echo "pilot done, evaluating $(date -Is)" >> "$STATUS"
run_one() {  # $1 script, $2 ckpt, $3 name
  [ -f "runs/phase_b2/multistart_$3.npz" ] && return 0
  while [ "$(free -g | awk 'NR==2{print $7}')" -lt 25 ]; do echo "MEMORY_ABORT $3 $(date -Is)" >> "$STATUS"; sleep 120; done
  echo "eval $3 $(date -Is)" >> "$STATUS"
  uv run "$1" "$2" "$3" > "runs/phase_b2/multistart_$3.log" 2>&1 \
    && echo "eval $3 OK $(tail -n 2 runs/phase_b2/multistart_$3.log | head -n 1) $(date -Is)" >> "$STATUS" \
    || echo "eval $3 FAILED $(date -Is)" >> "$STATUS"
}
run_one paper/phase_b/eval_rollout_multistart.py data/cnext_ckpt anchor
for seed in 1337 1338 1339; do
  for v in vanilla mlp30 mr30ryd; do
    run_one paper/phase_b/eval_rollout_multistart.py "runs/phase_b2/${v}_s${seed}.pt" "${v}_s${seed}"
  done
  run_one paper/phase_b/eval_rollout_multistart_noent.py "runs/phase_b2/noent_s${seed}.pt" "noent_s${seed}"
done
echo "MULTISTART_DONE $(date -Is)" >> "$STATUS"
sync
