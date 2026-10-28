#!/bin/bash
# §48j (doc 35, registered before any result): temporal X-point heads screen - Mxp vs Vxp (t20 recipe) on the 4 X-point coordinates, §47 folds;
# advance iff Vxp nsse <= 0.85 x Mxp on >= 2/3 folds; stop after fold 0 if > 0.95. One job at a time, after §48i.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48j.log
until grep -qE "^S48I_(DONE|FAIL)" $L/s48i.log 2>/dev/null; do sleep 120; done
echo "[s48j] start $(date -Is)" >> $LOG
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=24"
MJ='{"target": "xp"}'
VJ='{"target": "xp", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
run() { [ -f $L/quota/$1_f$3.npz ] && return 0
  env $E runs/capped.sh s48j$1$3 36 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "$2" $3 3 > $L/s48j_$1_f$3.log 2>&1
  echo "[s48j] $1 f$3 exit $? $(date -Is) $(grep ^QF_CV $L/s48j_$1_f$3.log)" >> $LOG; }
nsse() { grep "^QF_CV $1 fold $2/" $L/s48j_$1_f$2.log | grep -o "nsse [0-9.e+-]*" | cut -d' ' -f2; }
pass=0
for f in 0 1 2; do
  run Mxp "$MJ" $f; run Vxp "$VJ" $f
  m=$(nsse Mxp $f); v=$(nsse Vxp $f); [ -n "$m" ] && [ -n "$v" ] || { echo "S48J_FAIL fold $f missing output $(date -Is)" >> $LOG; exit 1; }
  r=$(awk -v v=$v -v m=$m 'BEGIN{printf "%.4f", v / m}'); echo "S48J_FOLD $f ratio Vxp/Mxp nsse $r $(date -Is)" >> $LOG
  awk -v r=$r 'BEGIN{exit !(r <= 0.85)}' && pass=$((pass + 1))
  [ "$f" = 0 ] && awk -v r=$r 'BEGIN{exit !(r > 0.95)}' && { echo "S48J_STOP fold 0 ratio $r > 0.95 $(date -Is)" >> $LOG; exit 0; }
  [ "$pass" -ge 2 ] && break
done
[ "$pass" -ge 2 ] && echo "S48J_ADVANCE $pass folds <= 0.85 $(date -Is)" >> $LOG || echo "S48J_STOP $pass/3 folds <= 0.85 $(date -Is)" >> $LOG
