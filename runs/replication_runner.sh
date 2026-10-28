#!/bin/bash
# Powered replication (31-powered-replication-prereg.md) then the RouteFinder EAL row. Starts after TDC_DONE
# (which itself waits for the HPO chain, so the ms2 configurations are frozen). One GPU job at a time; resumable.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
S=runs/replication/status
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 25 ]; do sleep 60; done; }
while ! grep -q TDC_DONE runs/tdc/status 2>/dev/null; do sleep 300; done
echo "replication runner start $(date -Is)" >> "$S"
# 1) Pelzer: 3 trunks x 4 adapter seeds = 12 units x 3 arms
mem_gate; echo "pelzer start $(date -Is)" >> "$S"
uv run --no-sync python paper/hpo/replicate.py pelzer --trunk-seeds 1337 1338 1339 --seeds 2001 2002 2003 2004 >> runs/replication/pelzer.log 2>&1 \
  && echo "REP_PELZER_DONE $(date -Is)" >> "$S" || echo "CRASH pelzer $(date -Is)" >> "$S"
# 2) Sophelio: 8 new seeds x 3 arms
mem_gate; echo "sophelio start $(date -Is)" >> "$S"
uv run --no-sync python paper/hpo/replicate.py sophelio --seeds 1340 1341 1342 1343 1344 1345 1346 1347 >> runs/replication/sophelio.log 2>&1 \
  && echo "REP_SOPHELIO_DONE $(date -Is)" >> "$S" || echo "CRASH sophelio $(date -Is)" >> "$S"
# 3) TokaMind: 4 new seeds x 3 arms, their warm-start protocol (same commands as runs/tokamind_runner.sh)
O=runs/tokamark; R=$O/results; L=$O/tokamind; PY=.venv-tokamark/bin/python; D=paper/tokamark/tokamind_hybrid.py
for seed in 1340 1341 1342 1343; do for v in vanilla mlp30 mr30ryd; do
  if [ ! -f $L/train_${v}_s${seed}.done ]; then
    mem_gate; echo "train tokamind $v s$seed $(date -Is)" >> "$S"
    $PY $D finetune --task task_1-1 --variant $v --seed $seed --workers 8 > $L/finetune_${v}_s${seed}.log 2>&1 \
      && touch $L/train_${v}_s${seed}.done || { echo "CRASH tokamind train $v s$seed $(date -Is)" >> "$S"; continue; }
  fi
  if [ ! -f $L/eval_${v}_s${seed}.done ]; then
    mem_gate; $PY $D eval --task task_1-1 --variant $v --seed $seed --workers 8 > $L/eval_${v}_s${seed}.log 2>&1 \
      && touch $L/eval_${v}_s${seed}.done && echo "result tokamind $v s$seed $(date -Is)" >> "$S" \
      || echo "CRASH tokamind eval $v s$seed $(date -Is)" >> "$S"
  fi
done; done; echo "REP_TOKAMIND_DONE $(date -Is)" >> "$S"
# 4) RouteFinder EAL (setting md): 3 arms x 3 seeds, only if the harness smoke passed
if [ -f runs/routefinder/EAL_SMOKE_OK ]; then
  for seed in 1 2 3; do for v in vanilla mlp30 mr30ryd; do
    [ -f runs/routefinder/eal_md_${v}_s${seed}.json ] && continue
    mem_gate; echo "rf-eal md $v s$seed $(date -Is)" >> "$S"
    (cd baseline/routefinder && ../../.venv-vrp/bin/python ../../paper/vrp/run_rf_eal.py $v --setting md --seed $seed) \
      >> runs/routefinder/eal_md.log 2>&1 && echo "result rf-eal md $v s$seed $(date -Is)" >> "$S" \
      || echo "CRASH rf-eal md $v s$seed $(date -Is)" >> "$S"
  done; done; echo "RF_EAL_DONE $(date -Is)" >> "$S"
else echo "SKIP rf-eal (no EAL_SMOKE_OK) $(date -Is)" >> "$S"; fi
echo "REPLICATION_RUNNER_DONE $(date -Is)" >> "$S"
