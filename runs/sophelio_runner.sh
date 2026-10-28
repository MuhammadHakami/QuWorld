#!/bin/bash
# Sophelio Ch1 (DIII-D) arms on the starter kit's UNetLite pipeline: wait for the dataset download, then
# {vanilla, mlp30, mr30ryd} x seeds {1337,1338,1339}, 200 shots (seed-42 sample + shot split), 50 epochs (their protocol).
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/sophelio; S=$O/status; mkdir -p $O; echo "runner start $(date -Is)" >> "$S"
while ! grep -q "\[dl\] sophelio done" runs/sophelio/download.log 2>/dev/null; do sleep 300; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
  [ -f $O/${v}_s${seed}.json ] && continue
  mem_gate; echo "arm $v s$seed $(date -Is)" >> "$S"
  uv run --no-sync python paper/sophelio/train_hybrid.py --variant $v --seed $seed --n-shots 200 --epochs 50 > $O/${v}_s${seed}.log 2>&1 || echo "CRASH $v s$seed $(date -Is)" >> "$S"
  [ -f $O/${v}_s${seed}.json ] && echo "result $v s$seed $(python3 -c "import json; d=json.load(open('$O/${v}_s${seed}.json')); o=d['official']; print(f\"S={o['S']:.4f} r2psi={o['r2_psi']:.4f} r2qb={o['r2_qb']:.4f} dlcfs={o['dlcfs']:.4f} cons={o['consistency']:.4f} testR2={d['test_R2']:.4f}\")")" >> "$S"
done; done
echo "SOPHELIO_DONE $(date -Is)" >> "$S"
