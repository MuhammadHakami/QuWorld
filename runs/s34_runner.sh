#!/bin/bash
# doc 35 §34/§34a: Flowers MHD_64 from scratch (authors' schedule/sampling, no clipping), low priority: only with >= 60 GB
# free and no MHD campaign job. A throwaway timing chunk (tag _timing) sets the epoch budget; the real run starts at step 0.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 THROTTLE_MS=15
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; S=runs/alloc/status; O=runs/phase_b_mhd; L=$O/flower_s1337.log; E=$O/flower_s1337.epochs
free_ok() { [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 60 ] && ! pgrep -f "capped.sh s23mhd " >/dev/null; }
if [ ! -f $E ]; then
  until free_ok; do sleep 300; done
  QW_GPU_GB=40 timeout 3000 runs/capped.sh s34t 60 uv run --no-sync python paper/phase_b/train_flowers_mhd.py 1337 1200 100 _timing >> $O/flower_timing.log 2>&1
  steps=$(grep -o "chunk done step=[0-9]*" $O/flower_timing.log | tail -1 | grep -o "[0-9]*$"); secs=$(grep -o "secs=[0-9]*" $O/flower_timing.log | tail -1 | grep -o "[0-9]*$")
  [ -n "$steps" ] && [ "$steps" -gt 0 ] || { echo "CRASH s34 timing chunk $(date -Is)" >> $S; exit 1; }
  proj=$(( 23100 * secs / steps )); ep=100; [ $proj -gt 604800 ] && ep=25
  echo "S34 timing: $steps steps in ${secs}s, 100 epochs projected ${proj}s -> $ep epochs $(date -Is)" >> $S
  rm -f $O/flower_timing_s1337.pt $O/flower_timing_s1337_latest.pt; echo $ep > $E
fi
ep=$(cat $E)
for attempt in $(seq 1 400); do
  grep -q target_reached $L 2>/dev/null && break
  free_ok || { sleep 300; continue; }
  QW_GPU_GB=40 timeout 3600 runs/capped.sh s34 60 uv run --no-sync python paper/phase_b/train_flowers_mhd.py 1337 2400 $ep >> $L 2>&1; sync
done
grep -q target_reached $L && echo "S34_TRAIN_DONE $(date -Is)" >> $S || echo "S34 INCOMPLETE $(date -Is)" >> $S
