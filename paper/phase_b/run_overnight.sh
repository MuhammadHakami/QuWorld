#!/bin/bash
# Phase-B overnight queue: fine-tune from the official checkpoint under the
# official protocol, then ONE final test evaluation per best-on-valid model,
# then paired Wilcoxon vs the reproduced anchor (runs/eval_official_cnext.csv).
#
# Usage: bash paper/phase_b/run_overnight.sh [budget_seconds_per_run]
# Default 7200s (2h) x {vanilla, mr30ryd, mlp30} x {1337,1338,1339}
# + mr30hea seed 1337 = 10 runs ~ 20h. Logs stream val_check lines for
# mid-run monitoring (paper/check_runs.py "phase_b/ft_").

set -u
cd "$(dirname "$0")/../.."
BUDGET=${1:-7200}

for seed in 1337 1338 1339; do
  for v in vanilla mr30ryd mlp30; do
    log="runs/phase_b/ft_${v}_s${seed}.log"
    [ -f "runs/phase_b/${v}_s${seed}.pt" ] && grep -q "^checkpoint:" "$log" 2>/dev/null && \
      { echo "skip ${v} s${seed} (done)"; continue; }
    echo "=== finetune ${v} seed ${seed} budget ${BUDGET}s ==="
    uv run paper/phase_b/finetune.py "$v" "$seed" "$BUDGET" > "$log" 2>&1 \
      || echo "CRASH ${v} s${seed}"
  done
done
# one extra digital-ansatz point
log="runs/phase_b/ft_mr30hea_s1337.log"
[ -f "runs/phase_b/mr30hea_s1337.pt" ] && grep -q "^checkpoint:" "$log" 2>/dev/null \
  || uv run paper/phase_b/finetune.py mr30hea 1337 "$BUDGET" > "$log" 2>&1 \
  || echo "CRASH mr30hea s1337"

echo "=== FINAL TEST EVALS (single shot per model, pre-registered) ==="
for ckpt in runs/phase_b/*.pt; do
  name=$(basename "$ckpt" .pt)
  csv="runs/phase_b/test_${name}.csv"
  [ -f "$csv" ] || uv run paper/phase_b/eval_official.py "$ckpt" "$csv" \
    > "runs/phase_b/test_${name}.log" 2>&1 || echo "EVAL_CRASH ${name}"
done

echo "=== PAIRED STATS vs anchor ==="
for csv in runs/phase_b/test_*.csv; do
  echo "--- $(basename "$csv") ---"
  uv run paper/phase_b/paired_stats.py "$csv" runs/eval_official_cnext.csv \
    || echo "STATS_CRASH $csv"
done
echo OVERNIGHT_DONE
