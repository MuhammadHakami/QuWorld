#!/bin/bash
# Fusion §41q FAST PROXY (user: faster iterations): J1 head arms on 1/3 of the TRAIN shots (STRAT_SUB=0/3) for 15 epochs (~1/12 of a
# full run), seeds 1 / 2, paired. Calibration first against the full-run verdicts (thz +0.00096, canon -0.0002, hfull +0.00007), then
# the untested arms (aux psi targets, canon + sign). After QHCS (mephiles). Read-out: paper/strat/fusion/proxy_report.py.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q -E "S37R_HEAD_DONE|Error" $L/QHCS_s1.log 2>/dev/null; do sleep 30; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1 STRAT_EPOCHS=15 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix STRAT_SUB=0/3"
for arm in ctrl: thz:STRAT_THZ=1 canon:STRAT_CANON=1 hfull:STRAT_HFULL=1 aux:STRAT_AUX_PSI=1 cs:STRAT_CANON=2; do
  n=${arm%%:*}; x=${arm#*:}
  for s in 1 2; do
    [ -f $L/head_boundary_v8_P${n}_s$s.npz ] && continue
    env $E $x STRAT_OUT_TAG=_P$n .venv-vrp/bin/python paper/strat/fusion/head_boundary.py $s > $L/P${n}_s$s.log 2>&1
  done
done
echo "PROXY_DONE $(date -Is)" >> $L/Pctrl_s1.log
