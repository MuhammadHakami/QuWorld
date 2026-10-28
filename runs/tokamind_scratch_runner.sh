#!/bin/bash
# TokaMind-Scratch trunk (their finetune_scratch.yaml recipe; paper TaskNRMSE 0.140 on task 1-1 = SoTA) + our arms.
# SoTA-first: vanilla s1337 must land near 0.140 before the arms run. Smoke on 8 shots first.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/tokamark; S=$O/status; R=$O/results; L=$O/tokamind; PY=.venv-tokamark/bin/python; D=paper/tokamark/tokamind_hybrid.py
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
arm() { # variant seed [subset]
  local v=$1 s=$2 sub=${3:+--subset $3} tag=${1}_s${2}_scratch${3:+_sub$3}
  [ -f $L/eval_$tag.done ] && return 0
  mem_gate; echo "train tokamind-scratch $tag $(date -Is)" >> $S
  QW_GPU_GB=40 runs/capped.sh tmscr 48 $PY $D finetune --init scratch --task task_1-1 --variant $v --seed $s $sub --workers 8 > $L/finetune_$tag.log 2>&1 || { echo "CRASH tokamind-scratch train $tag $(date -Is)" >> $S; return 1; }
  QW_GPU_GB=40 runs/capped.sh tmscr-e 48 $PY $D eval --init scratch --task task_1-1 --variant $v --seed $s $sub --workers 8 > $L/eval_$tag.log 2>&1 || { echo "CRASH tokamind-scratch eval $tag $(date -Is)" >> $S; return 1; }
  touch $L/eval_$tag.done
  echo "result tokamind-scratch $tag: $($PY -c "import json,sys; d=json.load(open('$R/tokamind_task_1-1_$tag.json')); print({k:{m:v for m,v in r.items() if 'nrmse' in m.lower()} for c in d['task_metrics'].values() for k,r in c.items()})" 2>&1 | tail -c 400) $(date -Is)" >> $S
}
arm vanilla 1337 8 || exit 1
arm vanilla 1337
for v in mr30ryd mlp30; do arm $v 1337; done
echo "TOKAMIND_SCRATCH_S1337_DONE $(date -Is)" >> $S
