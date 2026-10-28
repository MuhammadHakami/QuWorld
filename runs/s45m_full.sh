#!/bin/bash
# §45m full-size (runs only if the proxy CV passed): J1RLSF s1 + s2 = J1RL recipe + STRAT_SUBF=1, TRAIN only, concurrent; VT read-out
# paired vs J1RL s1 + s2 (all 743 VT shots, shot bootstrap). Rule: adopt iff mean-7 delta >= +0.0003 with CI > 0.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45m_full.log
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
echo "[s45m] start $(date -Is)" >> $LOG
for s in 1 2; do
  [ -f $L/head_boundary_v8_J1RLSF_s$s.pt ] || env $J STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_SUBF=1 \
    STRAT_OUT_TAG=_J1RLSF QW_GPU_GB=20 runs/capped.sh hJ1RLSF$s 40 .venv/bin/python paper/strat/fusion/head_boundary.py $s > $L/J1RLSF_s$s.log 2>&1 \
    && echo "J1RLSF_s${s}_TRAINED $(date -Is)" >> $LOG || echo "CRASH J1RLSF s$s $(date -Is)" >> $LOG &
done; wait
[ -f $L/head_boundary_v8_J1RLSF_s1.pt ] && [ -f $L/head_boundary_v8_J1RLSF_s2.pt ] || { echo "CRASH s45m heads missing $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s1+head_boundary_v8_J1RL_s2 head_boundary_v8_J1RLSF_s1+head_boundary_v8_J1RLSF_s2 >> $LOG 2>&1 || echo "CRASH vt shape $(date -Is)" >> $LOG
.venv/bin/python paper/strat/fusion/vt_eval.py shape head_boundary_v8_J1RL_s1 head_boundary_v8_J1RLSF_s1 >> $LOG 2>&1
echo "S45M_FULL_DONE $(date -Is)" >> $LOG
