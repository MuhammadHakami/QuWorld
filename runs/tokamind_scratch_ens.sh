#!/bin/bash
# Registered Sep 25 01:20 BEFORE any Scratch TEST score: equal-weight ensembles, no selection.
#   E1 = TokaMind-Scratch vanilla s1337 alone (the SoTA reproduction, paper 0.140)
#   E4 = mean of {Scratch vanilla s1337, FT-Base vanilla s1337/1338/1339} (diverse-trunk classical ensemble)
#   E4q = the same with the register arm when Scratch mr30ryd exists: {Scratch mr30ryd, FT-Base mr30ryd x3}
cd /home/mohhakami/Documents/paper/quworld; PY=.venv-tokamark/bin/python; D=paper/tokamark/tokamind_hybrid.py
S=runs/tokamark/status; P=runs/tokamark/preds; R=runs/tokamark/results; L=runs/tokamark/tokamind
gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 60; done; }
nrm() { .venv/bin/python -c "import json; d=json.load(open('$1'))['task_metrics']; print(list(d.values())[0]['task_1-1']['NRMSE_mean'])"; }
ens() { local v=$1 name=$2; local out=$R/tokamind_task_1-1_${v}_${name}.json; [ -f $out ] && return
  [ -f $P/${v}_s1337_scratch.npz ] || { gate; TOKAMIND_DUMP=$P/${v}_s1337_scratch.npz runs/capped.sh tm-dump 24 $PY $D eval --init scratch --task task_1-1 --variant $v --seed 1337 --workers 8 > runs/tokamark/dump_${v}_scratch.log 2>&1 || { echo "CRASH tokamind scratch dump $v $(date -Is)" >> $S; return; }; }
  gate; TOKAMIND_ENSEMBLE=$P/${v}_s1337_scratch.npz,$P/${v}_s1337.npz,$P/${v}_s1338.npz,$P/${v}_s1339.npz TOKAMIND_OUT=$out \
    runs/capped.sh tm-ens 24 $PY $D eval --init scratch --task task_1-1 --variant $v --seed 1337 --workers 8 > runs/tokamark/ens_${v}_${name}.log 2>&1 \
    && echo "result tokamind-scratch $v $name: $(nrm $out) $(date -Is)" >> $S || echo "CRASH tokamind $name $v $(date -Is)" >> $S; }
while [ ! -f $L/eval_vanilla_s1337_scratch.done ]; do sleep 120; done
ens vanilla E4
while [ ! -f $L/eval_mr30ryd_s1337_scratch.done ] && pgrep -x -f "/bin/bash runs/tokamind_scratch_runner.sh" > /dev/null; do sleep 300; done
[ -f $L/eval_mr30ryd_s1337_scratch.done ] && ens mr30ryd E4
[ -f $L/eval_mlp30_s1337_scratch.done ] && ens mlp30 E4
echo "TOKAMIND_SCRATCH_ENS_DONE $(date -Is)" >> $S
