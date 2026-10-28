#!/bin/bash
# §47 smoke: each backbone arm runs end to end (1 epoch, fold 0) - timing / memory only, no read-out used.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_smoke.log
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=1 QW_GPU_GB=40"
for a in 'smokeTg {"backbone":"tcnext"}' 'smokeHg {"backbone":"hope"}'; do set -- $a
  env $E runs/capped.sh s47smoke 50 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "$2" 0 3 >> $LOG 2>&1; echo "[smoke] $1 exit $? $(date -Is)" >> $LOG; done
echo S47_SMOKE_DONE >> $LOG
