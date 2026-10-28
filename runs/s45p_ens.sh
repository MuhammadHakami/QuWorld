#!/bin/bash
# §45p: mixed-loss vs same-loss 2-member ensembles on all VT shots (ens_eval.py), under vt.lock (one VT read-out at a time; builds run alone).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45p.log
echo "[s45p] start $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45p 40 bash -c '
  .venv/bin/python paper/strat/fusion/ens_eval.py shape head_boundary_v8_J1RL_s1+head_boundary_v8_J1RL_s2 head_boundary_v8_J1RL_s1+head_boundary_v8_J1RLH_s1
  .venv/bin/python paper/strat/fusion/ens_eval.py qb head_qb_QHR2c_s1+head_qb_QHR2_s1 head_qb_QHR2c_s1+head_qb_QHR2Hu_s1' >> $LOG 2>&1 || echo "CRASH s45p $(date -Is)" >> $LOG
echo "S45P_DONE $(date -Is)" >> $LOG
