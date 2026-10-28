#!/bin/bash
# §47 V smoke: whole-shot VisionHOPE arm end to end (1 epoch, fold 0) - timing / memory only, no read-out used.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_vsmoke.log
env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=1 QW_GPU_GB=40 \
  runs/capped.sh s47vsmoke 50 .venv/bin/python paper/strat/fusion/quota_fusion.py cv smokeVj '{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"mixer":64,"bb_depth":2,"tokens":8,"vh_pre":"stage1","jepa":100,"epochs":1}' 0 3 >> $LOG 2>&1
echo "[vsmoke] exit $? $(date -Is)" >> $LOG
