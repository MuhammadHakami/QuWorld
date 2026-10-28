#!/bin/bash
# SRR Stage B PILOT (Sep 23; non-decisive, pre-declared in 32-dynamic-register-configuration.md §7 addendum):
# TRL2D one-step, official CNextU-net, Phase B protocol (batch 16, lr 1e-4, AdamW, VALID every 180 s) but 5k steps,
# seed 1337, arms run one at a time: vanilla (length-matched), srr-lin, srr-mlp, srr-ryd. TEST is not touched here.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 PHASEB_OUT=runs/srr_pilot QW_GPU_GB=20
S=runs/srr_pilot/status; echo "pilot start $(date -Is)" >> "$S"
for v in vanilla srr-lin srr-mlp srr-ryd; do
  log=runs/srr_pilot/ft_${v}_s1337.log
  until grep -q "target_reached" "$log" 2>/dev/null; do
    runs/capped.sh srr-$v 24 .venv/bin/python paper/phase_b/finetune.py "$v" 1337 2400 1e-4 5000 >> "$log" 2>&1 \
      || { echo "CRASH $v $(date -Is)" >> "$S"; sleep 60; }
    grep -q "val_vrmse_1" "$log" || break
  done
  echo "done $v: $(grep -E 'val_vrmse_1|num_steps' "$log" | tail -2 | tr '\n' ' ') $(date -Is)" >> "$S"
done
echo "PILOT_DONE $(date -Is)" >> "$S"
