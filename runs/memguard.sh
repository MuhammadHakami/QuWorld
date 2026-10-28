#!/bin/bash
# v5 (Oct 12 01:22, after the 01:11 OOM reboot: MemAvailable 39 -> 0 GB within one earlyoom interval; v4 logged nothing - its kill
# path calls nvidia-smi AFTER memory is low, and nvidia-smi stalls under memory starvation). v5: the GPU-holder list is refreshed
# every ~10 s while memory is healthy (nvidia-smi under a 5 s timeout) and cached; when MemAvailable < LIMIT the kill path uses only
# /proc and the cache (no driver calls), logs BEFORE killing, and polls every 1 s. Order: cached GPU holders by GPU memory, then the
# largest-RSS python. Never the Phase-C teacher / Physics-IQ jobs (v2 invariant).
# v5.1 (Oct 12 09:44): Learn2Design / JAX processes are killed FIRST - they are not allowed on the Spark (lane split) and were the
# root cause of the 01:11, 01:49 and 02:43 collapses (JAX GPU preallocation in the shared unified memory).
#   start: setsid nohup bash runs/memguard.sh 20 &        log: runs/memguard.log
LIMIT=${1:-20}; LOG=/home/mohhakami/Documents/paper/quworld/runs/memguard.log
PROTECT='teacher_cache\.py|p1_train\.py|physiq_runner\.py'
CACHE=/dev/shm/memguard_gpu_pids; : > $CACHE
echo "memguard v5 start limit=${LIMIT}G pid=$$ $(date -Is)" >> "$LOG"
t=0
while :; do
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$avail" -lt "$LIMIT" ]; then
    py=$(ps -u "$(id -u)" -o pid=,rss=,comm= | awk '$3 ~ /^(python|pt_main_thread)/ {print $2, $1}' | sort -rn | awk '{print $2}')
    l2d=$(for v in $py; do tr '\0' ' ' < /proc/$v/cmdline 2>/dev/null | grep -Eq "Learn2Design|paper/l2d|jax|differometor" && echo $v; done)
    for v in $l2d $(cat $CACHE) $py; do
      cmd=$(tr '\0' ' ' < /proc/$v/cmdline 2>/dev/null)
      [ -z "$cmd" ] && continue
      echo "$cmd" | grep -Eq "$PROTECT" && continue
      echo "KILL pid=$v avail=${avail}G cmd=$(echo "$cmd" | cut -c1-200) $(date -Is)" >> "$LOG"
      kill -9 "$v" 2>/dev/null; sleep 5; break
    done
  elif [ $((t % 10)) -eq 0 ]; then
    timeout 5 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' '$2 ~ /^[0-9]+$/ {print $2, $1}' | sort -rn | awk '{print $2}' > $CACHE.tmp && mv $CACHE.tmp $CACHE
  fi
  t=$((t + 1)); sleep 1
done
