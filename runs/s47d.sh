#!/bin/bash
# §47d: q95 / betaN screen, Mqb vs Vqb on the §47 folds (starts when V s3 of §47c has finished training).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_screen.log
until grep -q 'V s3 exit' $L/s47c.log; do sleep 60; done
VQ='{"target":"qb","backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}'
run() { [ -f $L/quota/$1_f$3.npz ] && return 0
  env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=30 \
    runs/capped.sh s47$1$3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "$2" $3 3 > $L/s47_$1_f$3.log 2>&1
  echo "[s47] $1 f$3 exit $? $(date -Is) $(grep ^QF_CV $L/s47_$1_f$3.log)" >> $LOG; }
for f in 0 1 2; do run Mqb '{"target":"qb"}' $f & run Vqb "$VQ" $f; wait; done
.venv/bin/python paper/strat/fusion/quota_blend.py Mqb Vqb >> $LOG 2>&1
echo "S47D_DONE $(date -Is)" >> $LOG
