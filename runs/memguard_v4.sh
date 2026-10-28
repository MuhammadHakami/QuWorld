#!/bin/bash
# v4 (Sep 25 21:30): cheap, restartable jobs first (ADR-C runs and cache builds, 1-epoch RouteFinder pilots, RRNCO
# tests), then everything else by GPU memory; v3 killed a 76-min RouteFinder full run first (largest GPU holder).
# v3 (Sep 23, after two OOM reboots; replaces v2's fixed kill order). Jobs launched from Claude inherit oom_score_adj
# -1000, and GPU (unified) allocations on the GB10 are not charged to cgroup caps, so neither the kernel nor earlyoom
# stops a runaway job. When MemAvailable < LIMIT GiB, SIGKILL the job holding the most GPU memory (nvidia-smi), falling
# back to the largest-RSS python; v2's invariant is kept: never the Phase-C teacher / Physics-IQ jobs.
#   start: setsid nohup bash runs/memguard.sh 14 &        log: runs/memguard.log
LIMIT=${1:-14}; LOG=/home/mohhakami/Documents/paper/quworld/runs/memguard.log
PROTECT='teacher_cache\.py|p1_train\.py|physiq_runner\.py'
echo "memguard v4 start limit=${LIMIT}G pid=$$ $(date -Is)" >> "$LOG"
CHEAP='paper/alloc/adrc(_cache)?\.py|run_rf_eal\.py .* --epochs 1 |rrnco.*test\.py'
cands() {   # pids: cheap restartable jobs first (biggest GPU first), then all by GPU memory, then python RSS
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$2 ~ /^[0-9]+$/ {print $2, $1}' | sort -rn | awk '{print $2}' \
    | while read v; do tr '\0' ' ' < /proc/$v/cmdline 2>/dev/null | grep -Eq "$CHEAP" && echo $v; done
  ps -u "$(id -u)" -o pid=,comm=,args= | awk '$2 ~ /^(python|pt_main)/' | grep -E "$CHEAP" | awk '{print $1}'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$2 ~ /^[0-9]+$/ {print $2, $1}' | sort -rn | awk '{print $2}'
  ps -u "$(id -u)" -o pid=,rss=,comm= | awk '$3 ~ /^(python|pt_main_thread)/ {print $2, $1}' | sort -rn | awk '{print $2}'
}
while :; do
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$avail" -lt "$LIMIT" ]; then
    for v in $(cands); do
      cmd=$(tr '\0' ' ' < /proc/$v/cmdline 2>/dev/null)
      [ -z "$cmd" ] && continue
      echo "$cmd" | grep -Eq "$PROTECT" && continue
      echo "KILL pid=$v avail=${avail}G cmd=$(echo "$cmd" | cut -c1-200) $(date -Is)" >> "$LOG"
      kill -9 "$v" 2>/dev/null; sleep 10; break
    done
  fi
  sleep 2
done
