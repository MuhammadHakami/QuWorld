#!/bin/bash
# Session-surviving, freeze-resilient pilot. Launch:
export PATH="$HOME/.local/bin:$PATH"
#   setsid nohup bash paper/phase_b/run_pilot_detached.sh > runs/phase_b/pilot_runner.log 2>&1 < /dev/null &
# Also relaunched automatically at boot via crontab @reboot -> resume_campaign.sh.
# Each arm trains to TARGET_STEPS total, resuming from *_latest.pt after any
# kill (session teardown, earlyoom, cgroup cap, board latch-off + reboot).

set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export THROTTLE_MS=15          # shave sustained GPU load (board latch-off risk)
TARGET=${1:-3000}
BUDGET=1800
STATUS=runs/phase_b/pilot_status
MIN_FREE_GB=25

run_capped() {
  # Hard cgroup cap: the process dies cleanly long before the device can wedge.
  if systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 "$@"
  else
    "$@"
  fi
}

echo "pilot start $(date -Is) target=${TARGET} throttle=${THROTTLE_MS}ms" >> "$STATUS"
for v in vanilla mr30ryd mlp30; do
  log="runs/phase_b/pilot_${v}.log"
  for attempt in 1 2 3 4 5 6; do
    grep -q "target_reached" "$log" 2>/dev/null && break
    avail=$(free -g | awk 'NR==2{print $7}')
    if [ "$avail" -lt "$MIN_FREE_GB" ]; then
      echo "MEMORY_ABORT ${v} attempt ${attempt}: only ${avail}G free $(date -Is)" >> "$STATUS"
      sleep 120; continue
    fi
    echo "arm ${v} attempt ${attempt} $(date -Is) free=${avail}G" >> "$STATUS"
    run_capped uv run paper/phase_b/finetune.py "$v" 1337 "$BUDGET" 1e-4 "$TARGET" >> "$log" 2>&1
    sync
  done
  grep -q "target_reached" "$log" 2>/dev/null \
    && echo "arm ${v} DONE $(date -Is)" >> "$STATUS" \
    || echo "arm ${v} FAILED after 6 attempts $(date -Is)" >> "$STATUS"
done
echo "PILOT_DONE $(date -Is)" >> "$STATUS"
sync
