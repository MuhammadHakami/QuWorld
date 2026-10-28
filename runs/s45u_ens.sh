#!/bin/bash
# §45u: canonical twins as extra ensemble members on normal-Ip shots (reversed shots identical in both arms via route:), all VT shots, under vt.lock.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45u.log
echo "[s45u] start $(date -Is)" >> $LOG
B="route:head_boundary_v8_J1RL_s1+head_boundary_v8_J1RL_s2|head_boundary_v8_J1RLCS_s1"; C="route:head_boundary_v8_J1RL_s1+head_boundary_v8_J1RLCS_s1|head_boundary_v8_J1RLCS_s1"
QB="route:head_qb_QHR2c_s1+head_qb_QHR2_s1|head_qb_QHR2CS_s1"; QC="route:head_qb_QHR2c_s1+head_qb_QHR2CS_s1|head_qb_QHR2CS_s1"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45u 40 bash -c ".venv/bin/python paper/strat/fusion/ens_eval.py shape '$B' '$C'; .venv/bin/python paper/strat/fusion/ens_eval.py qb '$QB' '$QC'" >> $LOG 2>&1 || echo "CRASH s45u $(date -Is)" >> $LOG
echo "S45U_DONE $(date -Is)" >> $LOG
