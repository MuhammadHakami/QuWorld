#!/bin/bash
# Optuna HPO campaign (user directive Sep 21): per benchmark, every arm gets the same trial budget on TRAIN/VALID,
# then the selected configuration runs 3 seeds ONCE on TEST. Order = cheapest rows first. Resumable (SQLite studies,
# final JSONs skipped if present). GPU co-runner: waits for MemAvailable >= 30 GB (no video generation running).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; export PATH="$HOME/.local/bin:$PATH"
S=runs/hpo/status; mkdir -p runs/hpo; echo "hpo runner start $(date -Is)" >> "$S"
gate() { while :; do a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  pgrep -f "physiq_runner[.]py --stage generate" > /dev/null && { sleep 120; continue; }
  [ "$a" -lt 30 ] && { sleep 60; continue; }; break; done; }
hpo() { bench=$1; arm=$2; trials=$3
  if [ ! -f runs/hpo/$bench/${arm}_final.json ]; then
    gate; echo "search $bench $arm trials=$trials $(date -Is)" >> "$S"
    uv run --no-sync python paper/hpo/driver.py $bench $arm search --trials $trials > runs/hpo/${bench}_${arm}_search.log 2>&1 || { echo "CRASH search $bench $arm $(date -Is)" >> "$S"; return 1; }
    echo "best $bench $arm: $(grep -E 'search done' runs/hpo/${bench}_${arm}_search.log | tail -1 | cut -c1-200) $(date -Is)" >> "$S"
    gate; echo "final $bench $arm $(date -Is)" >> "$S"
    uv run --no-sync python paper/hpo/driver.py $bench $arm final > runs/hpo/${bench}_${arm}_final.log 2>&1 || { echo "CRASH final $bench $arm $(date -Is)" >> "$S"; return 1; }
    echo "result $bench $arm: $(grep -E 'FINAL' runs/hpo/${bench}_${arm}_final.log | tail -1 | cut -c1-160) $(date -Is)" >> "$S"
  fi; }
for arm in none mlp q; do hpo torax $arm 16; done; echo "HPO_TORAX_DONE $(date -Is)" >> "$S"
for arm in none mlp q; do hpo pelzer $arm 12; done; echo "HPO_PELZER_DONE $(date -Is)" >> "$S"
for arm in none mlp q; do hpo sophelio $arm 10; done; echo "HPO_SOPHELIO_DONE $(date -Is)" >> "$S"
echo "HPO_DONE $(date -Is)" >> "$S"
