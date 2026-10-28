#!/bin/bash
# Wave G (29-graph-configuration-plan.md): register graph bake-off on the TRL2D arena.
# Stage 1: 12 configs + matched controls + reference trio at seed 1337. Stage 2: top-3 by val VRMSE@8 (+controls)
# at seeds 1338/1339. Fixed 2000 steps, TD k=4. GPU co-runner: waits for MemAvailable >= 40 GB and no
# Physics-IQ generate process. Idempotent (skips logs that already have val_vrmse_8).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
O=runs/waveG; S=$O/status; mkdir -p $O; echo "waveG runner start $(date -Is)" >> "$S"
gate() { while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  pgrep -f "physiq_runner[.]py --stage generate" > /dev/null && { sleep 120; continue; }
  pgrep -f "wmb/generate[.]py" > /dev/null && { sleep 120; continue; }
  [ "$a" -lt 40 ] && { sleep 60; continue; }; break; done; }
run() { v=$1; seed=$2; L=$O/${v}_s${seed}.log
  [ -f "$L" ] && grep -q "val_vrmse_8" "$L" && return 0
  gate; echo "train $v s$seed $(date -Is)" >> "$S"
  .venv/bin/python paper/train.py "$v" "$seed" 100000 2000 4 > "$L" 2>&1 || { echo "CRASH $v s$seed $(date -Is)" >> "$S"; return 1; }
  echo "result $v s$seed: $(grep -E 'val_vrmse_1|val_vrmse_8|num_params|train_seconds|participation' "$L" | tr -s ' ' | tr '\n' ' ') $(date -Is)" >> "$S"; }
CONFIGS="q30-geo=chain q30-geo=ring q30-geo=tri q30-sp=0.7 q30-sp=1.5 q30-part=6x5 q30-part=2x15 q30-link=cascade q30-link=shared q30-gate q30-gate-link=cascade q30-geo=tri-sp=0.7"
# Stage 1 ------------------------------------------------------------------------------------------
for v in q30 mlp30 none; do run $v 1337; done
for v in $CONFIGS; do run $v 1337; run ${v}-mlp 1337; done
echo "WAVEG_STAGE1_DONE $(date -Is)" >> "$S"
# Stage 2: top-3 configs by val VRMSE@8 at seed 1337 (ties: lower VRMSE@1) ------------------------
TOP3=$(python3 - << 'PY'
import re, glob
rows = []
for f in glob.glob("runs/waveG/q30-*_s1337.log"):
    v = f.split("/")[-1][:-len("_s1337.log")]
    if v.endswith("-mlp"): continue
    t = open(f).read(); m8 = re.search(r"val_vrmse_8:\s+([0-9.]+)", t); m1 = re.search(r"val_vrmse_1:\s+([0-9.]+)", t)
    if m8 and m1: rows.append((float(m8.group(1)), float(m1.group(1)), v))
rows.sort(); print(" ".join(v for _, _, v in rows[:3]))
PY
)
echo "stage2 configs: $TOP3 $(date -Is)" >> "$S"
for seed in 1338 1339; do for v in $TOP3; do run $v $seed; run ${v}-mlp $seed; done; done
for seed in 1338 1339; do for v in q30 mlp30 none; do [ -f runs/ag15_${v/q30/mr30ryd}_k4_s${seed}.log ] || run $v $seed; done; done
python3 paper/summarize_waveG.py > $O/summary.md 2>&1
echo "WAVEG_DONE $(date -Is)" >> "$S"
