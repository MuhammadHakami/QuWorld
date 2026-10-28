#!/bin/bash
# §47 screen (registered 2026-10-17T13:41:17, arms amended: VisionHOPE only): arms M / V (Hg fold 0 diagnostic) x 3 stratified shot folds of half of TRAIN, 20 epochs, seed 1; two at a time.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_screen.log
echo "[s47] start $(date -Is)" >> $LOG
job() {  # arm json fold
  [ -f $L/quota/$1_f$3.npz ] && return 0
  env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=30 \
    runs/capped.sh s47$1$3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "$2" $3 3 > $L/s47_$1_f$3.log 2>&1
  echo "[s47] $1 f$3 exit $? $(date -Is) $(grep ^QF_CV $L/s47_$1_f$3.log)" >> $LOG
}
export -f job; export L LOG
# one job per line "arm|json|fold" (no xargs quote processing of the JSON)
printf '%s\n' 'V|{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}|0' 'M|{}|1' 'V|{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}|1' 'M|{}|2' 'V|{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}|2' \
  | xargs -d '\n' -P 2 -n 1 bash -c 'IFS="|" read -r a j f <<< "$0"; job "$a" "$j" "$f"'
.venv/bin/python paper/strat/fusion/quota_blend.py M V Hg >> $LOG 2>&1
echo "S47_SCREEN_DONE $(date -Is)" >> $LOG
