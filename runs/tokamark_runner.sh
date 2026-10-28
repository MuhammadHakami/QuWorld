#!/bin/bash
# TokaMark task_1-1 (equilibrium reconstruction from magnetics, official random split) — arms on THEIR CNN baseline
# (tokamark_baseline MultiConv_MLP, their trainer/evaluator/metrics, config_model.yaml hyper-parameters, seed 23):
# variant in {vanilla, mlp30, mr30ryd}; waits for the full shot download; memory gate 20 GB; status runs/tokamark/status.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/tokamark; S=$O/status; mkdir -p $O; echo "runner start $(date -Is)" >> "$S"
while ! grep -qE "\[tokamark-dl\] done: ok [0-9]+ fail 0 " runs/tokamark/download_shots.log 2>/dev/null; do sleep 300; done
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 20 ]; do sleep 60; done; }
CFG=/home/mohhakami/Documents/paper/quworld/paper/tokamark/config_local.yaml
for seed in 23; do for v in vanilla mlp30 mr30ryd; do
  R=$O/results/random/cnn_${v}/task_1-1/seed_${seed}
  if [ ! -f $O/train_${v}_s${seed}.done ]; then
    mem_gate; echo "train $v s$seed $(date -Is)" >> "$S"
    .venv-tokamark/bin/python paper/tokamark/train_hybrid.py --task task_1-1 --config $CFG --seed $seed --variant $v --validate_every 100 > $O/train_${v}_s${seed}.log 2>&1 && touch $O/train_${v}_s${seed}.done || echo "CRASH train $v s$seed $(date -Is)" >> "$S"
  fi
  if [ -f $O/train_${v}_s${seed}.done ] && [ ! -d "$R" ]; then
    mem_gate; echo "eval $v s$seed $(date -Is)" >> "$S"
    .venv-tokamark/bin/python paper/tokamark/eval_hybrid.py --task task_1-1 --config $CFG --seed $seed --variant $v > $O/eval_${v}_s${seed}.log 2>&1 || echo "CRASH eval $v s$seed $(date -Is)" >> "$S"
    echo "result $v s$seed: $(ls $R 2>/dev/null | tr '\n' ' ')" >> "$S"
  fi
done; done
echo "TOKAMARK_DONE $(date -Is)" >> "$S"
