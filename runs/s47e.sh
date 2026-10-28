#!/bin/bash
# §47e: Vt (VisionHOPE-T stage-1 blocks) and Vd (dynamic 3-D tethered register) vs V on the §47 folds; starts after the §47c VT read.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_screen.log
until grep -q '^S47C_' $L/s47c.log; do sleep 60; done
VT='{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"mixer":64,"bb_depth":2,"tokens":8,"vh_pre":"stage1"}'
VD='{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8,"reg":"tether3d"}'
run() { [ -f $L/quota/$1_f$3.npz ] && return 0
  env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=30 \
    runs/capped.sh s47$1$3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "$2" $3 3 > $L/s47_$1_f$3.log 2>&1
  echo "[s47] $1 f$3 exit $? $(date -Is) $(grep ^QF_CV $L/s47_$1_f$3.log)" >> $LOG; }
for f in 0 1 2; do run Vt "$VT" $f & run Vd "$VD" $f; wait; done
.venv/bin/python paper/strat/fusion/quota_blend.py M V Vt Vd >> $LOG 2>&1
.venv/bin/python paper/strat/fusion/quota_blend.py V Vt Vd >> $LOG 2>&1
echo "S47E_DONE $(date -Is)" >> $LOG
