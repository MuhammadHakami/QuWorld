#!/bin/bash
# Pre-registered Gutzwiller (mean-field) control, 2026-09-06 (see 18-gutzwiller-control-plan.md):
# arena pilot (3 seeds x 2,000 fixed steps, k=4) then official-protocol TD arms
# (3 seeds x 10k steps, identical to the mr30ryd/noent arms), then hand the GPU
# back to the resumable MHD campaign. Reboot-safe: resume_campaign.sh relaunches
# this chain while GUTZ_DONE is absent; the lock prevents a second live copy.
# Launch: setsid nohup bash /home/mohhakami/Documents/paper/quworld/paper/phase_b/run_gutz.sh \
#   >> /home/mohhakami/Documents/paper/quworld/runs/phase_b2/gutz_runner.log 2>&1 < /dev/null &
export PATH="$HOME/.local/bin:$PATH"
set -u
cd /home/mohhakami/Documents/paper/quworld
exec 9> runs/phase_b2/gutz.lock
flock -n 9 || { echo "another run_gutz.sh holds the lock; exiting"; exit 0; }
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True THROTTLE_MS=15
STATUS=runs/phase_b2/gutz_status
MIN_FREE_GB=25

run_capped() {
  if systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 "$@"
  else
    "$@"
  fi
}
wait_for_memory() {  # $1 = tag for the status line
  while [ "$(free -g | awk 'NR==2{print $7}')" -lt "$MIN_FREE_GB" ]; do
    echo "MEMORY_ABORT $1 $(date -Is)" >> "$STATUS"; sleep 120
  done
}

echo "gutz start $(date -Is)" >> "$STATUS"
# the GPU is ours only when neither the MHD runner shell nor its trainer is alive
while pgrep -f "finetune_mhd.py|run_mhd_campaign.sh" > /dev/null; do sleep 30; done
echo "gpu free $(date -Is)" >> "$STATUS"

for seed in 1337 1338 1339; do
  log="runs/ladder_gutz30_k4_s${seed}.log"
  grep -q "val_vrmse_8" "$log" 2>/dev/null && continue
  wait_for_memory "pilot s${seed}"
  echo "pilot gutz30 k4 s${seed} $(date -Is)" >> "$STATUS"
  run_capped uv run paper/train.py gutz30 "$seed" 1500 2000 4 > "$log" 2>&1 \
    || echo "PILOT_CRASH s${seed} $(date -Is)" >> "$STATUS"
done
echo "PILOT_DONE $(date -Is)" >> "$STATUS"

for seed in 1337 1338 1339; do
  log="runs/phase_b2/td_gutz30_s${seed}.log"
  for attempt in $(seq 1 12); do
    grep -q "target_reached" "$log" 2>/dev/null && break
    wait_for_memory "arm s${seed}"
    echo "arm gutz30 s${seed} attempt ${attempt} $(date -Is)" >> "$STATUS"
    run_capped uv run paper/phase_b/finetune_td.py gutz30 "$seed" 2400 1e-4 10000 4 >> "$log" 2>&1
    sync
  done
  grep -q "target_reached" "$log" 2>/dev/null \
    && echo "arm gutz30 s${seed} DONE $(date -Is)" >> "$STATUS" \
    || echo "arm gutz30 s${seed} INCOMPLETE $(date -Is)" >> "$STATUS"
done
echo "GUTZ_DONE $(date -Is)" >> "$STATUS"
# hand the GPU back: the MHD campaign skips finished arms and resumes partial ones
setsid nohup bash /home/mohhakami/Documents/paper/quworld/paper/phase_b/run_mhd_campaign.sh \
  >> runs/phase_b_mhd/mhd_runner.log 2>&1 < /dev/null &
echo "MHD relaunched $(date -Is)" >> "$STATUS"
sync
