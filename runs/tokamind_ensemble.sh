#!/bin/bash
# TokaMark 1-1 seed ensembles (Sep 23, classical lift, labelled as ensembles): record each fine-tune's decoded test
# predictions (TOKAMIND_DUMP), then score the 3-seed mean per arm with TokaMind's own metric code (TOKAMIND_ENSEMBLE).
# Record/replay verified: a replayed 1-member ensemble reproduces the member's NRMSE_mean exactly (0.3726833, sub8).
cd /home/mohhakami/Documents/paper/quworld; PY=.venv-tokamark/bin/python; D=paper/tokamark/tokamind_hybrid.py
S=runs/tokamark/status; mkdir -p runs/tokamark/preds
gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
for v in vanilla mlp30 mr30ryd; do for s in 1337 1338 1339; do
  f=runs/tokamark/preds/${v}_s${s}.npz; [ -f $f ] && continue; gate
  TOKAMIND_DUMP=$f runs/capped.sh tm-dump 24 $PY $D eval --task task_1-1 --variant $v --seed $s --workers 8 > runs/tokamark/dump_${v}_s${s}.log 2>&1 \
    || echo "CRASH tokamind dump $v s$s $(date -Is)" >> "$S"
done; done
for v in vanilla mlp30 mr30ryd; do
  out=runs/tokamark/results/tokamind_task_1-1_${v}_ens3.json; [ -f $out ] && continue; gate
  TOKAMIND_ENSEMBLE=runs/tokamark/preds/${v}_s1337.npz,runs/tokamark/preds/${v}_s1338.npz,runs/tokamark/preds/${v}_s1339.npz TOKAMIND_OUT=$out \
    runs/capped.sh tm-ens 24 $PY $D eval --task task_1-1 --variant $v --seed 1337 --workers 8 > runs/tokamark/ens_${v}.log 2>&1 \
    && echo "result tokamind ens3 $v: $(.venv/bin/python -c "import json; d=json.load(open('$out'))['task_metrics']; print(list(d.values())[0]['task_1-1']['NRMSE_mean'])") $(date -Is)" >> "$S" \
    || echo "CRASH tokamind ens3 $v $(date -Is)" >> "$S"
done
echo "TOKAMIND_ENS_DONE $(date -Is)" >> "$S"
