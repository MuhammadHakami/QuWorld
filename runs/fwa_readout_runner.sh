#!/bin/bash
# FW-A (Sep 19 sweep): readout families on the TRL2D arena — QREADOUT in {z, diag, cond2} x {mlp30, mr30ryd} x 3 seeds,
# round-3 protocol (paper/train.py, 300 s sprint, val vrmse@1 / vrmse@8). The MLP control is re-matched under each
# readout automatically (its budget is _count() of a reference register built under the same QREADOUT).
# Chained behind TOKAMIND_DONE (last of the benchmark lanes); 14 GB memory gate; sentinel runs/fwa/FWA_DONE.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/fwa; S=$O/status; mkdir -p $O; echo "fwa runner start $(date -Is)" >> "$S"
while ! grep -q "^TOKAMIND_DONE" runs/tokamark/status 2>/dev/null; do sleep 600; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 14 ]; do sleep 60; done; }
for seed in 1337 1338 1339; do for ro in z diag cond2; do for v in mlp30 mr30ryd; do
  L=$O/${ro}_${v}_s${seed}.log; [ -f "$L" ] && grep -q "val_vrmse_8" "$L" && continue
  mem_gate; echo "train $ro $v s$seed $(date -Is)" >> "$S"
  QREADOUT=$ro .venv/bin/python paper/train.py $v $seed 300 > "$L" 2>&1 || echo "CRASH $ro $v s$seed $(date -Is)" >> "$S"
  echo "result $ro $v s$seed: $(grep -E 'val_vrmse_1|val_vrmse_8|num_params|num_steps' "$L" | tr -s ' ' | tr '\n' ' ') $(date -Is)" >> "$S"
done; done; done
python3 - <<'PY' > $O/summary.md
import glob, re, statistics as st, collections
rows = collections.defaultdict(list)
for f in sorted(glob.glob("runs/fwa/*_s*.log")):
    t = open(f).read(); m = {k: float(re.search(rf"{k}:\s+([0-9.]+)", t).group(1)) for k in ("val_vrmse_1", "val_vrmse_8", "num_params") if re.search(rf"{k}:\s+([0-9.]+)", t)}
    if len(m) == 3: rows[re.sub(r"_s\d+\.log$", "", f.split("/")[-1])].append(m)
print("| readout_variant | n | vrmse@1 mean±sd | vrmse@8 mean±sd | params |\n|---|---|---|---|---|")
for k, ms in rows.items():
    v1 = [m["val_vrmse_1"] for m in ms]; v8 = [m["val_vrmse_8"] for m in ms]
    sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
    print(f"| {k} | {len(ms)} | {st.mean(v1):.4f}±{sd(v1):.4f} | {st.mean(v8):.4f}±{sd(v8):.4f} | {int(ms[0]['num_params'])} |")
PY
echo "FWA_DONE $(date -Is)" >> "$S"
