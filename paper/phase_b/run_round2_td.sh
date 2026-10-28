#!/bin/bash
# Phase-B ROUND 2: TD-objective checkpoint surgery — the wave-1.5-informed rematch.
export PATH="$HOME/.local/bin:$PATH"
# Chained behind OVERNIGHT_DONE by resume_campaign or a watcher. Absolute paths.
# 3 arms x 3 seeds @ 10k TD steps (each TD step ~ K fwd passes), then single-shot
# test evals + paired stats vs the anchor.
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True THROTTLE_MS=15
STATUS=runs/phase_b2/round2_status
TARGET=${1:-10000}
mkdir -p runs/phase_b2
echo "round2 start $(date -Is) target=${TARGET} k=4" >> "$STATUS"

# --- GAP JOBS first (minutes, unblock Gates B and C before the long arms) ---
if [ ! -f runs/eval_official_cnext_mhd.csv ]; then
  echo "gapjob mhd_anchor start $(date -Is)" >> "$STATUS"
  WELL_DATASET=MHD_64 EVAL_BATCH=8 systemd-run --user --scope -q -p MemoryMax=80G \
    uv run paper/phase_b/eval_official.py data/cnext_mhd_ckpt runs/eval_official_cnext_mhd.csv \
    > runs/phase_b_mhd_anchor.log 2>&1 \
    && echo "gapjob mhd_anchor DONE $(date -Is)" >> "$STATUS" \
    || echo "gapjob mhd_anchor FAILED $(date -Is)" >> "$STATUS"
fi
if [ -d data/phase_c/renders ] && [ ! -f data/phase_c/latents/.done ]; then
  echo "gapjob vae_encode start $(date -Is)" >> "$STATUS"
  systemd-run --user --scope -q -p MemoryMax=80G \
    uv run paper/phase_c/vae_encode.py > runs/phase_c_vae.log 2>&1 \
    && { mkdir -p data/phase_c/latents; touch data/phase_c/latents/.done; \
         echo "gapjob vae_encode DONE $(date -Is)" >> "$STATUS"; } \
    || echo "gapjob vae_encode FAILED $(date -Is)" >> "$STATUS"
fi

for seed in 1337 1338 1339; do
  for v in vanilla mr30ryd mlp30; do
    log="runs/phase_b2/td_${v}_s${seed}.log"
    for attempt in $(seq 1 20); do
      grep -q "target_reached" "$log" 2>/dev/null && break
      avail=$(free -g | awk 'NR==2{print $7}')
      [ "$avail" -lt 25 ] && { echo "MEMORY_ABORT ${v} s${seed} $(date -Is)" >> "$STATUS"; sleep 120; continue; }
      echo "arm ${v} s${seed} attempt ${attempt} $(date -Is)" >> "$STATUS"
      systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 \
        uv run paper/phase_b/finetune_td.py "$v" "$seed" 2400 1e-4 "$TARGET" 4 >> "$log" 2>&1
      sync
    done
    if grep -q "target_reached" "$log" 2>/dev/null; then
      echo "arm ${v} s${seed} DONE $(date -Is)" >> "$STATUS"
      # immediate single-shot test eval (same discipline, earlier verdict)
      ckpt="runs/phase_b2/${v}_s${seed}.pt"
      csv="runs/phase_b2/test_${v}_s${seed}.csv"
      [ -f "$ckpt" ] && [ ! -f "$csv" ] && {
        systemd-run --user --scope -q -p MemoryMax=80G \
          uv run paper/phase_b/eval_official.py "$ckpt" "$csv" \
          > "runs/phase_b2/test_${v}_s${seed}.log" 2>&1 \
          || echo "EVAL_CRASH ${v} s${seed}" >> "$STATUS"
        [ -f "$csv" ] && {
          { echo "--- td_${v}_s${seed} $(date -Is) ---";
            uv run paper/phase_b/paired_stats.py "$csv" runs/eval_official_cnext.csv; } \
            >> runs/phase_b2/paired_stats_report.txt 2>&1
          echo "paired-stats td_${v}_s${seed} appended $(date -Is)" >> "$STATUS"
        }
      }
    else
      echo "arm ${v} s${seed} INCOMPLETE $(date -Is)" >> "$STATUS"
    fi
  done
done
echo "=== single-shot test evals ===" >> "$STATUS"
for ckpt in runs/phase_b2/*_s*.pt; do
  case "$ckpt" in *_latest.pt) continue;; esac
  name=$(basename "$ckpt" .pt)
  csv="runs/phase_b2/test_${name}.csv"
  [ -f "$csv" ] && continue
  uv run paper/phase_b/eval_official.py "$ckpt" "$csv" > "runs/phase_b2/test_${name}.log" 2>&1 \
    || echo "EVAL_CRASH ${name}" >> "$STATUS"
done
for csv in runs/phase_b2/test_*.csv; do
  { echo "--- $(basename "$csv") ---";
    uv run paper/phase_b/paired_stats.py "$csv" runs/eval_official_cnext.csv; } \
    >> runs/phase_b2/paired_stats_report.txt 2>&1
done
echo "ROUND2_DONE $(date -Is)" >> "$STATUS"
sync
