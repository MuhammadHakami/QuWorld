#!/bin/bash
# Memory flight recorder (Oct 12, after two GPU-memory collapses): every 5 s, MemAvailable and per-process GPU memory -> runs/memlog.csv.
LOG=/home/mohhakami/Documents/paper/quworld/runs/memlog.csv
while :; do
  a=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  g=$(timeout 3 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' | tr -d ' ')
  echo "$(date +%H:%M:%S),$a,$g" >> $LOG
  sleep 5
done
