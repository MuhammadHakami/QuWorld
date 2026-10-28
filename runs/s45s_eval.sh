#!/bin/bash
# §45s read-out after the twins: VT router read for the register routing + label-free / X-point checks (s45s_eval.py), under vt.lock.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45s.log
until grep -q "^S45S_TRAINED" $LOG; do sleep 120; done
QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QH="head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
B="route:head_boundary_v8_J1RL_s1*24+$QPI+$QH|head_boundary_v8_J1RLCS_s1*24+$QPI+$QH"; C="head_boundary_v8_J1RLCS_s1*24+$QPI+$QHCS"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s45sev 40 bash -c ".venv/bin/python paper/strat/fusion/router_eval.py shape '$B' '$C'; .venv/bin/python paper/strat/fusion/s45s_eval.py" >> $LOG 2>&1 || echo "CRASH s45s eval $(date -Is)" >> $LOG
echo "S45S_DONE $(date -Is)" >> $LOG
