#!/bin/bash
# doc 35 §23: MHD_64, same campaign recipe as the published neck models (finetune_mhd.py, 20000 steps, lr 1e-4,
# resumable 2400 s chunks, THROTTLE_MS=15, 80 GB cap, >= 25 GB free), seed 1337, then single-shot TEST.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 THROTTLE_MS=15
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; S=runs/alloc/status; OUT=runs/phase_b_mhd
until grep -q S23_TRL2D_DONE $S; do sleep 120; done
for v in q30-geo=3d q30-geo=3d-mlp; do log=$OUT/ft_${v}_s1337.log
  for attempt in $(seq 1 30); do
    grep -q target_reached "$log" 2>/dev/null && break
    [ "$(free -g | awk 'NR==2{print $7}')" -lt 25 ] && { sleep 120; continue; }
    QW_GPU_GB=28 timeout 3600 runs/capped.sh s23mhd 80 uv run --no-sync python paper/phase_b/finetune_mhd.py "$v" 1337 2400 1e-4 20000 >> "$log" 2>&1; sync
  done
  grep -q target_reached "$log" && echo "S23 mhd $v DONE $(date -Is)" >> $S || echo "S23 mhd $v INCOMPLETE $(date -Is)" >> $S
done; echo "S23_MHD_DONE $(date -Is)" >> $S
