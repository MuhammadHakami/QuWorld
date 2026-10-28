#!/bin/bash
# Phase-B MHD_64 campaign, freeze-resilient — mirror of run_overnight_detached.sh.
# DO NOT launch while another GPU campaign is live. Launch (after audit GO):
#   setsid nohup bash paper/phase_b/run_mhd_campaign.sh > runs/phase_b_mhd/mhd_runner.log 2>&1 < /dev/null &
# Every arm is step-targeted and resumable (finetune_mhd.py *_latest.pt);
# test set touched exactly once per final model.
export PATH="$HOME/.local/bin:$PATH"  # @reboot/cron PATH lacks ~/.local/bin (uv) — see 92a7d33

set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export THROTTLE_MS=15
TARGET=${1:-20000}
BUDGET=2400
OUT=runs/phase_b_mhd
STATUS=$OUT/mhd_status
ANCHOR_CSV=runs/eval_official_cnext_mhd.csv
MIN_FREE_GB=25

mkdir -p "$OUT"

run_capped() {
  if systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 "$@"
  else
    "$@"
  fi
}

echo "mhd campaign start $(date -Is) target=${TARGET}" >> "$STATUS"

ARMS="vanilla:1337 mr30ryd:1337 mlp30:1337 vanilla:1338 mr30ryd:1338 mlp30:1338 vanilla:1339 mr30ryd:1339 mlp30:1339"
for arm in $ARMS; do
  v=${arm%%:*}; seed=${arm##*:}
  log="$OUT/ft_${v}_s${seed}.log"
  for attempt in $(seq 1 20); do
    grep -q "target_reached" "$log" 2>/dev/null && break
    avail=$(free -g | awk 'NR==2{print $7}')
    if [ "$avail" -lt "$MIN_FREE_GB" ]; then
      echo "MEMORY_ABORT ${v} s${seed}: ${avail}G free $(date -Is)" >> "$STATUS"
      sleep 120; continue
    fi
    echo "arm ${v} s${seed} attempt ${attempt} $(date -Is) free=${avail}G" >> "$STATUS"
    run_capped uv run paper/phase_b/finetune_mhd.py "$v" "$seed" "$BUDGET" 1e-4 "$TARGET" >> "$log" 2>&1
    sync
  done
  grep -q "target_reached" "$log" 2>/dev/null \
    && echo "arm ${v} s${seed} DONE $(date -Is)" >> "$STATUS" \
    || echo "arm ${v} s${seed} INCOMPLETE $(date -Is)" >> "$STATUS"
done

echo "=== single-shot test evals $(date -Is) ===" >> "$STATUS"

# Anchor first (paired-stats baseline; README_mhd.md expects mean_vrmse ~0.1633).
if [ ! -f "$ANCHOR_CSV" ]; then
  WELL_DATASET=MHD_64 EVAL_BATCH=8 run_capped uv run paper/phase_b/eval_official.py \
    data/cnext_mhd_ckpt "$ANCHOR_CSV" > "$OUT/eval_anchor.log" 2>&1 \
    || echo "EVAL_CRASH anchor" >> "$STATUS"
  sync
fi

for ckpt in "$OUT"/*_s*.pt; do
  case "$ckpt" in *_latest.pt) continue;; esac
  name=$(basename "$ckpt" .pt)
  csv="$OUT/test_${name}.csv"
  [ -f "$csv" ] && continue
  # eval_official.load_model rebuilds hybrid .pt checkpoints via
  # build_hybrid(variant), whose default model_dir is the TRL2D checkpoint
  # (README_mhd.md port note 3). Patch the default to the MHD checkpoint
  # in-process instead of editing the live eval script.
  WELL_DATASET=MHD_64 EVAL_BATCH=8 run_capped uv run python -c "
import sys
sys.path.insert(0, 'paper/phase_b')
import cnext_hybrid
_orig = cnext_hybrid.build_hybrid
_mhd = str(cnext_hybrid.ROOT / 'data/cnext_mhd_ckpt')
cnext_hybrid.build_hybrid = (
    lambda variant, model_dir=_mhd, q_layers=4:
        _orig(variant, model_dir=model_dir, q_layers=q_layers))
import eval_official
eval_official.main()
" "$ckpt" "$csv" > "$OUT/test_${name}.log" 2>&1 || echo "EVAL_CRASH ${name}" >> "$STATUS"
  sync
done

echo "=== paired stats vs MHD anchor $(date -Is) ===" >> "$STATUS"
for csv in "$OUT"/test_*.csv; do
  { echo "--- $(basename "$csv") ---";
    uv run paper/phase_b/paired_stats.py "$csv" "$ANCHOR_CSV"; } \
    >> "$OUT/paired_stats_report.txt" 2>&1
done

echo "MHD_CAMPAIGN_DONE $(date -Is)" >> "$STATUS"
sync
