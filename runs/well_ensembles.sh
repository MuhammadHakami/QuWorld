#!/bin/bash
# The Well TRL2D one-step TEST: 3-seed ensembles of the Phase B fine-tunes per arm (classical lift, labelled ensemble;
# arms stay matched: each is the mean of its own 3 seeds). Inference only; waits for >= 30 GB free.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=16
S=runs/phase_b/ensemble_status
gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
for v in vanilla mlp30 mr30ryd; do
  out=runs/phase_b/test_${v}_ens3.csv; [ -f $out ] && continue; gate
  m="runs/phase_b/${v}_s1337.pt+runs/phase_b/${v}_s1338.pt+runs/phase_b/${v}_s1339.pt"
  runs/capped.sh well-ens 24 .venv/bin/python paper/phase_b/eval_official.py "$m" "$out" > runs/phase_b/ens_${v}.log 2>&1 \
    && echo "result TRL2D ens3 $v: $(grep mean_vrmse runs/phase_b/ens_${v}.log) $(date -Is)" >> "$S" || echo "CRASH TRL2D ens3 $v $(date -Is)" >> "$S"
done
echo "WELL_ENS_DONE $(date -Is)" >> "$S"
