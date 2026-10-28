#!/bin/bash
# doc 35 §38ac after stage 1: stage-2 quantum NAS on the best stage-1 classical config, then the full retrain of the best
# config (60 epochs, all TRAIN) and the VALID-sample head screen vs J1 (single seed).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while ! grep -q "S38AC_HPO_DONE" $L/s38ac_hpo.log 2>/dev/null; do sleep 60; done
best() { .venv/bin/python -c "
import optuna, sys; st = optuna.load_study(study_name='$1', storage='sqlite:///runs/strat/fusion/hpo_qpikan.db'); b = st.best_trial; print(b.number, b.value)"; }
read B1 V1 < <(best qpikan_v1); echo "S38AC stage1 best trial $B1 $V1 $(date -Is)" >> $L/s38ac_hpo.log
flock runs/strat/gpu.lock env QW_GPU_GB=40 runs/capped.sh s38ac 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --stage2 $B1 --study qpikan_q2 \
  --trials 16 --frac 0.25 --epochs 6 >> $L/s38ac_q2.log 2>&1
read B2 V2 < <(best qpikan_q2); echo "S38AC stage2 best trial $B2 $V2 $(date -Is)" >> $L/s38ac_q2.log
if python3 -c "import sys; sys.exit(0 if float('$V2') > float('$V1') else 1)"; then S=qpikan_q2; T=$B2; else S=qpikan_v1; T=$B1; fi
flock runs/strat/gpu.lock env QW_GPU_GB=40 runs/capped.sh s38ac 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --final $T --study $S --epochs 60 \
  --seed 1 >> $L/s38ac_final.log 2>&1
.venv/bin/python paper/strat/fusion/head_screen.py head_boundary_v8_J1_s1 head_boundary_v8_QK${T}_s1 "head_boundary_v8_J1_s1+head_boundary_v8_QK${T}_s1" >> $L/s38ac_final.log 2>&1
echo "S38AC_FINAL_DONE study $S trial $T $(date -Is)" >> $L/s38ac_final.log
