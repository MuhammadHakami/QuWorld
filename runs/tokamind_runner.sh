#!/bin/bash
# TokaMark task_1-1 on the SoTA trunk (TokaMind FT-Base, paper 0.150 TaskNRMSE): THEIR warm-start protocol
# (5 ep heads + 15 ep full, AdamW, batch 512, patience 10) with our register wrapped on the CLS latent.
# Chain: waits for the CNN-lane sentinel TOKAMARK_DONE (=> shots local, one benchmark co-runner next to the teacher),
# memory gate 20 GB; smoke on 8 shots first; then {vanilla, mlp30, mr30ryd} x seeds {1337,1338,1339}; status lines
# in runs/tokamark/status; sentinel TOKAMIND_DONE.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/tokamark; S=$O/status; R=$O/results; L=$O/tokamind; mkdir -p $R $L
PY=.venv-tokamark/bin/python; D=paper/tokamark/tokamind_hybrid.py
echo "tokamind runner start $(date -Is)" >> "$S"
while ! grep -q "^TOKAMARK_DONE" "$S" 2>/dev/null; do sleep 300; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
nrmse() { $PY - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); out = []
for csv, rows in d.get("task_metrics", {}).items():
    for task, m in rows.items():
        for k, v in m.items():
            if "nrmse" in k.lower(): out.append(f"{task}:{k}={v:.4f}" if isinstance(v, float) else f"{task}:{k}={v}")
print(" ".join(out) if out else "NO_METRICS")
PY
}
# smoke: 8 shots, quantum arm, both stages, must produce task metrics
if [ ! -f $L/SMOKE_OK ]; then
  mem_gate; echo "train tokamind smoke mr30ryd s1337 sub8 $(date -Is)" >> "$S"
  $PY $D finetune --task task_1-1 --variant mr30ryd --seed 1337 --subset 8 --workers 4 > $L/smoke_finetune.log 2>&1 \
   && $PY $D eval --task task_1-1 --variant mr30ryd --seed 1337 --subset 8 --workers 4 > $L/smoke_eval.log 2>&1 \
   && [ "$(nrmse $R/tokamind_task_1-1_mr30ryd_s1337_sub8.json)" != "NO_METRICS" ] \
   && { touch $L/SMOKE_OK; echo "result tokamind smoke: $(nrmse $R/tokamind_task_1-1_mr30ryd_s1337_sub8.json) $(date -Is)" >> "$S"; } \
   || { echo "CRASH tokamind smoke $(date -Is) BLOCKED" >> "$S"; exit 1; }
fi
for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
  J=$R/tokamind_task_1-1_${v}_s${seed}.json
  if [ ! -f $L/train_${v}_s${seed}.done ]; then
    mem_gate; echo "train tokamind $v s$seed $(date -Is)" >> "$S"
    $PY $D finetune --task task_1-1 --variant $v --seed $seed --workers 8 > $L/finetune_${v}_s${seed}.log 2>&1 \
      && touch $L/train_${v}_s${seed}.done || { echo "CRASH tokamind train $v s$seed $(date -Is)" >> "$S"; continue; }
  fi
  if [ ! -f $L/eval_${v}_s${seed}.done ]; then
    mem_gate; echo "eval tokamind $v s$seed $(date -Is)" >> "$S"
    $PY $D eval --task task_1-1 --variant $v --seed $seed --workers 8 > $L/eval_${v}_s${seed}.log 2>&1 \
      && touch $L/eval_${v}_s${seed}.done && echo "result tokamind $v s$seed: $(nrmse $J) $(date -Is)" >> "$S" \
      || echo "CRASH tokamind eval $v s$seed $(date -Is)" >> "$S"
  fi
done; done
echo "TOKAMIND_DONE $(date -Is)" >> "$S"
