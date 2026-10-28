#!/bin/bash
# SRR pilot: the length-matched vanilla arm was SIGKILLed by earlyoom at 03:02 (CatBoost GPU reservation); re-run it after
# the three SRR arms, same protocol (seed 1337, 5k steps, PHASEB_OUT=runs/srr_pilot).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 PHASEB_OUT=runs/srr_pilot QW_GPU_GB=20
S=runs/srr_pilot/status
until grep -q PILOT_DONE "$S"; do sleep 120; done
rm -f runs/srr_pilot/vanilla_s1337.pt runs/srr_pilot/vanilla_s1337_latest.pt; mv runs/srr_pilot/ft_vanilla_s1337.log runs/srr_pilot/ft_vanilla_s1337.killed.log
log=runs/srr_pilot/ft_vanilla_s1337.log
until grep -q "target_reached" "$log" 2>/dev/null; do
  runs/capped.sh srr-vanilla 24 .venv/bin/python paper/phase_b/finetune.py vanilla 1337 2400 1e-4 5000 >> "$log" 2>&1 || { echo "CRASH vanilla-redo $(date -Is)" >> "$S"; sleep 60; }
done
echo "done vanilla (redo): $(grep -E 'val_vrmse_1|num_steps' "$log" | tail -2 | tr '\n' ' ') $(date -Is)" >> "$S"; echo "PILOT_COMPLETE $(date -Is)" >> "$S"
