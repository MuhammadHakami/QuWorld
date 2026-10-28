#!/bin/bash
# Wave G addendum (pre-registered 05:25 Sep 21, 29-graph-configuration-plan.md): qubit-scarcity pressure on the
# participation gate. GATE_PENALTY in {0.03, 0.1, 0.3} x q30-gate, seed 1337, fixed 2000 steps TD k=4; the ungated
# control is the Stage-1 q30 / q30-gate-mlp. Chained behind WAVEG_DONE; same GPU gate as the main runner.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
O=runs/waveG; S=$O/status
while ! grep -q "^WAVEG_DONE" "$S" 2>/dev/null; do sleep 300; done
gate() { while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  pgrep -f "physiq_runner[.]py --stage generate" > /dev/null && { sleep 120; continue; }
  pgrep -f "wmb/generate[.]py" > /dev/null && { sleep 120; continue; }
  [ "$a" -lt 40 ] && { sleep 60; continue; }; break; done; }
for lam in 0.03 0.1 0.3; do
  L=$O/q30-gate-lam${lam}_s1337.log; [ -f "$L" ] && grep -q "val_vrmse_8" "$L" && continue
  gate; echo "train q30-gate lam=$lam s1337 $(date -Is)" >> "$S"
  GATE_PENALTY=$lam .venv/bin/python paper/train.py q30-gate 1337 100000 2000 4 > "$L" 2>&1 || { echo "CRASH q30-gate lam=$lam $(date -Is)" >> "$S"; continue; }
  echo "result q30-gate lam=$lam s1337: $(grep -E 'val_vrmse_1|val_vrmse_8|participation|gate_penalty' "$L" | tr -s ' ' | tr '\n' ' ') $(date -Is)" >> "$S"
done
echo "WAVEG_GATE_DONE $(date -Is)" >> "$S"
