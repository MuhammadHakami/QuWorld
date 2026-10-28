#!/bin/bash
# Wave 1.6 — E3 seed replication (3D conditioned geometry) + kmix hard-routing validation.
# Fixed 2,000 steps, k=1, seeds per arm below. Pre-registered at commit time.
export PATH="$HOME/.local/bin:$PATH"
set -u
cd /home/mohhakami/Documents/paper/quworld
export PYTHONUNBUFFERED=1
STATUS=runs/ag_status
echo "wave16 start $(date -Is)" >> "$STATUS"
JOBS="cg30-3d:1338 cg30-3d:1339 kmix30:1337 kmix30:1338 kmix30:1339"
for job in $JOBS; do
  v=${job%%:*}; s=${job##*:}
  log="runs/ag16_${v}_k1_s${s}.log"
  grep -q "val_vrmse_1" "$log" 2>/dev/null && continue
  echo "run16 ${v} s${s} $(date -Is)" >> "$STATUS"
  uv run paper/train.py "$v" "$s" 900 2000 1 > "$log" 2>&1 || echo "CRASH16 ${v} s${s}" >> "$STATUS"
done
echo "WAVE16_DONE $(date -Is)" >> "$STATUS"
