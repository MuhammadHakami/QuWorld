#!/bin/bash
# §47b (amended): Vjc = cold V + VisionHOPE-T stage-1 blocks + 1,500 JEPA steps (fold's training shots) before the supervised 20 epochs.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_screen.log
VJ='{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"mixer":64,"bb_depth":2,"tokens":8,"vh_pre":"stage1","jepa":1500}'
for f in 0 1 2; do [ -f $L/quota/Vjc_f$f.npz ] && continue
  env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=30 \
    runs/capped.sh s47Vjc$f 40 .venv/bin/python paper/strat/fusion/quota_fusion.py cv Vjc "$VJ" $f 3 > $L/s47_Vjc_f$f.log 2>&1
  echo "[s47] Vjc f$f exit $? $(date -Is) $(grep ^QF_CV $L/s47_Vjc_f$f.log)" >> $LOG; done
.venv/bin/python paper/strat/fusion/quota_blend.py M V Vjc >> $LOG 2>&1
.venv/bin/python paper/strat/fusion/quota_blend.py V Vjc >> $LOG 2>&1
echo "S47_JEPA_DONE $(date -Is)" >> $LOG
