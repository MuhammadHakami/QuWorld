#!/bin/bash
# SRR Stage B PILOT, runner v2 (Sep 23): same pre-declared protocol as srr_pilot_runner.sh (seed 1337, 5k steps, Phase B
# settings, VALID only). Each arm resumes from its *_latest.pt until `target_reached` (up to 6 attempts), so a guard kill
# costs at most one VALID interval. Starts only when the RealPDE trunk evaluation has finished (memory).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 PHASEB_OUT=runs/srr_pilot QW_GPU_GB=20
S=runs/srr_pilot/status
while ps -eo args | grep -q "[e]val_trunk.py --models"; do sleep 120; done
if [ -f runs/srr_pilot/ft_vanilla_s1337.log ] && ! grep -q target_reached runs/srr_pilot/ft_vanilla_s1337.log; then
  mv runs/srr_pilot/ft_vanilla_s1337.log runs/srr_pilot/ft_vanilla_s1337.killed.log; rm -f runs/srr_pilot/vanilla_s1337*.pt
fi
echo "pilot v2 start $(date -Is)" >> "$S"
for v in srr-lin srr-mlp srr-ryd vanilla; do
  log=runs/srr_pilot/ft_${v}_s1337.log
  for attempt in 1 2 3 4 5 6; do
    grep -q "target_reached" "$log" 2>/dev/null && break
    runs/capped.sh srr-$v 24 .venv/bin/python paper/phase_b/finetune.py "$v" 1337 2400 1e-4 5000 >> "$log" 2>&1 \
      || { echo "CRASH $v attempt $attempt $(date -Is)" >> "$S"; sleep 120; }
  done
  echo "done $v: $(grep -E 'val_vrmse_1|num_steps' "$log" | tail -2 | tr '\n' ' ') $(grep -c target_reached "$log") $(date -Is)" >> "$S"
done
echo "PILOT_DONE $(date -Is)" >> "$S"
