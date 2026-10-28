#!/bin/bash
# Optuna campaign, round 2 (Sep 22): ADAPTER arms re-tuned with a multi-seed objective (mean VALID over seeds
# 1337+1338), same trial budgets, then the selected config x 3 seeds once on TEST. Trunk results from round 1 stand.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
S=runs/hpo/status; echo "hpo ms2 runner start $(date -Is)" >> "$S"
gate() { while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  pgrep -f "physiq_runner[.]py --stage generate" > /dev/null && { sleep 120; continue; }
  [ "$a" -lt 30 ] && { sleep 60; continue; }; break; done; }
hpo() { bench=$1; arm=$2; trials=$3
  if [ ! -f runs/hpo/$bench/${arm}_ms2_final.json ]; then
    gate; echo "search-ms2 $bench $arm trials=$trials $(date -Is)" >> "$S"
    uv run --no-sync python paper/hpo/driver.py $bench $arm search --trials $trials --search-seeds 1337 1338 --study ms2 > runs/hpo/${bench}_${arm}_ms2_search.log 2>&1 || { echo "CRASH search-ms2 $bench $arm $(date -Is)" >> "$S"; return 1; }
    echo "best-ms2 $bench $arm: $(grep -E 'search done' runs/hpo/${bench}_${arm}_ms2_search.log | tail -1 | cut -c1-200) $(date -Is)" >> "$S"
    gate; echo "final-ms2 $bench $arm $(date -Is)" >> "$S"
    uv run --no-sync python paper/hpo/driver.py $bench $arm final --study ms2 > runs/hpo/${bench}_${arm}_ms2_final.log 2>&1 || { echo "CRASH final-ms2 $bench $arm $(date -Is)" >> "$S"; return 1; }
    echo "result-ms2 $bench $arm: $(grep -E 'FINAL' runs/hpo/${bench}_${arm}_ms2_final.log | tail -1 | cut -c1-160) $(date -Is)" >> "$S"
  fi; }
for arm in q mlp; do hpo torax $arm 16; done; echo "HPO_MS2_TORAX_DONE $(date -Is)" >> "$S"
for arm in q mlp; do hpo pelzer $arm 12; done; echo "HPO_MS2_PELZER_DONE $(date -Is)" >> "$S"
for arm in q mlp; do hpo sophelio $arm 10; done; echo "HPO_MS2_SOPHELIO_DONE $(date -Is)" >> "$S"
echo "HPO_MS2_DONE $(date -Is)" >> "$S"
