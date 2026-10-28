#!/bin/bash
# Doc 35 §37 MHD launcher: waits for an optional condition, then takes the shared GPU lock, re-checks MemAvailable >= 30 GB
# and runs one capped job (QW_GPU_GB=12, 30-minute timeout, 16 GB cgroup).
#   paper/strat/mhd/launch.sh <min_snaps> <python args...>
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
SN=/tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad/snaps
need=$1; shift
avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
until [ "$(ls $SN/flower_step*.pt 2>/dev/null | wc -l)" -ge "$need" ] && [ "$(avail)" -ge 30 ]; do sleep 30; done
exec 9>runs/strat/gpu.lock; flock 9
until [ "$(avail)" -ge 30 ]; do sleep 30; done
echo "launch $(date -Is) avail=$(avail)G: $*"
timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_mhd 16 uv run --no-sync python "$@"
echo "exit $? $(date -Is)"
