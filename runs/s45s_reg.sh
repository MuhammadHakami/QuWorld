#!/bin/bash
# §45s register part early: the VT router read (J1QHCS done 08:07) does not need the X-point twins, so it runs as soon as vt.lock frees
# (after §45t); s45s_eval2.sh then runs only the X-point / label-free read-out (s45s_eval.py) after S45S_TRAINED.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45s.log
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QH="head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
B="route:head_boundary_v8_J1RL_s1*24+$QPI+$QH|head_boundary_v8_J1RLCS_s1*24+$QPI+$QH"; C="head_boundary_v8_J1RLCS_s1*24+$QPI+$QHCS"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45sreg 40 .venv/bin/python paper/strat/fusion/router_eval.py shape "$B" "$C" >> $LOG 2>&1 || echo "CRASH s45s reg $(date -Is)" >> $LOG
echo "S45S_REG_DONE $(date -Is)" >> $LOG
