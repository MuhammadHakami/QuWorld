#!/bin/bash
# Pre-registered no-entanglement training ablation: 3 seeds, TD, product-state from step 0.
export PATH="$HOME/.local/bin:$PATH"
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True THROTTLE_MS=15
STATUS=runs/phase_b2/ablation_status
echo "ablation start $(date -Is)" >> "$STATUS"
for seed in 1337 1338 1339; do
  log="runs/phase_b2/td_noent_s${seed}.log"
  for attempt in $(seq 1 12); do
    grep -q "target_reached" "$log" 2>/dev/null && break
    avail=$(free -g | awk 'NR==2{print $7}')
    [ "$avail" -lt 25 ] && { echo "MEMORY_ABORT s${seed} $(date -Is)" >> "$STATUS"; sleep 120; continue; }
    echo "arm noent s${seed} attempt ${attempt} $(date -Is)" >> "$STATUS"
    systemd-run --user --scope -q -p MemoryMax=80G -p MemorySwapMax=0 \
      uv run paper/phase_b/finetune_td_ablate.py noent "$seed" 2400 1e-4 10000 4 >> "$log" 2>&1
    sync
  done
  grep -q "target_reached" "$log" 2>/dev/null \
    && echo "arm noent s${seed} DONE $(date -Is)" >> "$STATUS" \
    || echo "arm noent s${seed} INCOMPLETE $(date -Is)" >> "$STATUS"
done
echo "ABLATION_DONE $(date -Is)" >> "$STATUS"
