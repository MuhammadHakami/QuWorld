#!/bin/bash
# Full Phase-B campaign, freeze-resilient. Launch (after audit GO + healthy pilot):
export PATH="$HOME/.local/bin:$PATH"
#   setsid nohup bash paper/phase_b/run_overnight_detached.sh > runs/phase_b/overnight_runner.log 2>&1 < /dev/null &
# Auto-resumed at boot by resume_campaign.sh (crontab @reboot). Every arm is
# step-targeted and resumable; test set touched exactly once per final model.

set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export THROTTLE_MS=15
TARGET=${1:-20000}
BUDGET=2400
STATUS=runs/phase_b/overnight_status
MIN_FREE_GB=25

run_capped() {
  if systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 "$@"
  else
    "$@"
  fi
}

echo "overnight start $(date -Is) target=${TARGET}" >> "$STATUS"

ARMS="vanilla:1337 mr30ryd:1337 mlp30:1337 vanilla:1338 mr30ryd:1338 mlp30:1338 vanilla:1339 mr30ryd:1339 mlp30:1339 mr30hea:1337"
for arm in $ARMS; do
  v=${arm%%:*}; seed=${arm##*:}
  log="runs/phase_b/ft_${v}_s${seed}.log"
  for attempt in $(seq 1 20); do
    grep -q "target_reached" "$log" 2>/dev/null && break
    avail=$(free -g | awk 'NR==2{print $7}')
    if [ "$avail" -lt "$MIN_FREE_GB" ]; then
      echo "MEMORY_ABORT ${v} s${seed}: ${avail}G free $(date -Is)" >> "$STATUS"
      sleep 120; continue
    fi
    echo "arm ${v} s${seed} attempt ${attempt} $(date -Is) free=${avail}G" >> "$STATUS"
    run_capped uv run paper/phase_b/finetune.py "$v" "$seed" "$BUDGET" 1e-4 "$TARGET" >> "$log" 2>&1
    sync
  done
  grep -q "target_reached" "$log" 2>/dev/null \
    && echo "arm ${v} s${seed} DONE $(date -Is)" >> "$STATUS" \
    || echo "arm ${v} s${seed} INCOMPLETE $(date -Is)" >> "$STATUS"
done

echo "=== single-shot test evals $(date -Is) ===" >> "$STATUS"
for ckpt in runs/phase_b/*_s*.pt; do
  case "$ckpt" in *_latest.pt) continue;; esac
  name=$(basename "$ckpt" .pt)
  csv="runs/phase_b/test_${name}.csv"
  [ -f "$csv" ] && continue
  run_capped uv run paper/phase_b/eval_official.py "$ckpt" "$csv" \
    > "runs/phase_b/test_${name}.log" 2>&1 || echo "EVAL_CRASH ${name}" >> "$STATUS"
  sync
done

echo "=== paired stats vs anchor $(date -Is) ===" >> "$STATUS"
for csv in runs/phase_b/test_*.csv; do
  { echo "--- $(basename "$csv") ---";
    uv run paper/phase_b/paired_stats.py "$csv" runs/eval_official_cnext.csv; } \
    >> runs/phase_b/paired_stats_report.txt 2>&1
done

echo "OVERNIGHT_DONE $(date -Is)" >> "$STATUS"
sync
