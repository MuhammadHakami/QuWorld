#!/bin/bash
# §45w (doc 35): compute-neutral F17 - (2) XP1CS vs XP1 on VT normal / reversed frames (light, no lock); (1) whole-canonical register vs the
# adopted F17 router on all VT shots (ens_eval shape, vt.lock).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45w.log; F=paper/strat/fusion
echo "[s45w] start $(date -Is)" >> $LOG
QW_GPU_GB=8 runs/capped.sh s45wxp 12 .venv/bin/python $F/s45w_xp.py >> $LOG 2>&1 || echo "CRASH s45w xp $(date -Is)" >> $LOG
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QH="head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
QHCS=${QH//J1QH_/J1QHCS_}; N="head_boundary_v8_J1RL_s1*24"; R="head_boundary_v8_J1RLCS_s1*24"
B="route:$N+$QPI+$QH|$R+$QPI+$QHCS"; C="route:$N+$QPI+$QHCS|$R+$QPI+$QHCS"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45wreg 40 .venv/bin/python $F/ens_eval.py shape "$B" "$C" >> $LOG 2>&1 \
  && echo "S45W_DONE $(date -Is)" >> $LOG || echo "CRASH s45w reg $(date -Is)" >> $LOG
