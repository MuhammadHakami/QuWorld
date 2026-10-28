#!/bin/bash
# Launch a job as a preferred OOM victim (oom_score_adj +800; shells started by Claude are -1000 and pass it on), inside a
# memory-capped cgroup when the user systemd manager is up. GB10 unified memory: GPU allocations are NOT charged to the
# cgroup cap, so runs/memguard.sh (kills the biggest GPU job when MemAvailable is low) is the real backstop.
#   runs/capped.sh <name> <max_GiB> <command...>
name=$1; mem=$2; shift 2
# compile caches on disk, not /tmp (a reboot wiped /tmp/torchinductor_* on Oct 15 -> minutes of CPU-only recompilation at GPU 0%)
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.triton/cache}
if systemctl --user is-system-running >/dev/null 2>&1 || systemctl --user show-environment >/dev/null 2>&1; then
  exec choom -n 800 -- systemd-run --user --scope --quiet --unit="qw-${name}-$(date +%H%M%S)-$$" -p MemoryMax=${mem}G -p MemorySwapMax=0 "$@"
fi
exec choom -n 800 -- "$@"
