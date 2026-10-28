#!/bin/bash
# Wave G-k (29-graph-configuration-plan.md): per-input budget selector on the arena. Seed-1337 screen of
# k in {3,5,7} x {chosen, random-k control, MLP control}; promising k -> seeds 1338/1339. Same GPU gate.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
O=runs/waveG; S=$O/status; echo "waveG-k runner start $(date -Is)" >> "$S"
while ! grep -q "^TRANSFER_DONE" "$S" 2>/dev/null; do sleep 120; done
gate() { while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  pgrep -f "physiq_runner[.]py --stage generate" > /dev/null && { sleep 120; continue; }
  pgrep -f "wmb/generate[.]py" > /dev/null && { sleep 120; continue; }
  [ "$a" -lt 40 ] && { sleep 60; continue; }; break; done; }
run() { v=$1; seed=$2; L=$O/${v}_s${seed}.log
  [ -f "$L" ] && grep -q "val_vrmse_8" "$L" && return 0
  gate; echo "train $v s$seed $(date -Is)" >> "$S"
  .venv/bin/python paper/train.py "$v" "$seed" 100000 2000 4 > "$L" 2>&1 || { echo "CRASH $v s$seed $(date -Is)" >> "$S"; return 1; }
  echo "result $v s$seed: $(grep -E 'val_vrmse_1|val_vrmse_8|num_params|participation' "$L" | tr -s ' ' | tr '\n' ' ') $(date -Is)" >> "$S"; }
v8() { grep -oE "val_vrmse_8:\s+[0-9.]+" "$O/$1_s$2.log" 2>/dev/null | grep -oE "[0-9.]+$"; }
for k in 3 5 7; do run q30-k=$k 1337; run q30-k=$k-randk 1337; run q30-k=$k-mlp 1337; done
echo "WAVEGK_STAGE1_DONE $(date -Is)" >> "$S"
REF=$(v8 q30 1337)
for k in 3 5 7; do
  q=$(v8 q30-k=$k 1337); r=$(v8 q30-k=$k-randk 1337)
  if python3 -c "import sys; sys.exit(0 if float('$q') < float('$r') and float('$q') < float('$REF') else 1)"; then
    echo "promising k=$k (q $q < randk $r, < ref $REF) -> seeds 1338/1339 $(date -Is)" >> "$S"
    for seed in 1338 1339; do run q30-k=$k $seed; run q30-k=$k-randk $seed; run q30-k=$k-mlp $seed; done
  else echo "k=$k not promising (q $q, randk $r, ref $REF) $(date -Is)" >> "$S"; fi
done
echo "WAVEGK_DONE $(date -Is)" >> "$S"
