#!/bin/bash
# §48l (doc 35, registered before any result): t20 temporal recipe for q95 / betaN - Vqb2 vs Mqb (§47 folds); advance iff the optimal blend beats Mqb
# by >= +0.0005 q/b mean R2 on 3/3 folds (stop after a failing fold). After s48t.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48l.log
until grep -q "^S48T_DONE" $L/s48t.log 2>/dev/null; do sleep 120; done
echo "[s48l] start $(date -Is)" >> $LOG
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=30"
VJ='{"target": "qb", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
run() { [ -f $L/quota/$1_f$3.npz ] && return 0
  env $E runs/capped.sh s48l$1$3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "$2" $3 3 > $L/s48l_$1_f$3.log 2>&1
  echo "[s48l] $1 f$3 exit $? $(date -Is) $(grep ^QF_CV $L/s48l_$1_f$3.log)" >> $LOG; }
gain() { .venv/bin/python - "$1" <<'PY'
import sys, numpy as np
sys.path.insert(0, "paper/strat/fusion"); import quota_blend as QB
k = sys.argv[1]; Q = "runs/strat/fusion/quota/"; zc, za = np.load(Q + f"Mqb_f{k}.npz"), np.load(Q + f"Vqb2_f{k}.npz"); assert (zc["rows"] == za["rows"]).all()
s = QB.stats(zc["y"], zc["pred"], za["pred"]); print(f"{np.mean([t['r2_best'] - t['r2_ctrl'] for t in s]):+.5f} single {np.mean([t['r2_arm'] for t in s]):.5f} ctrl {np.mean([t['r2_ctrl'] for t in s]):.5f}")
PY
}
for f in 0 1 2; do
  run Mqb '{"target": "qb"}' $f & run Vqb2 "$VJ" $f; wait
  g=$(gain $f 2>>$LOG); echo "S48L_FOLD $f blend gain $g $(date -Is)" >> $LOG
  awk -v g="${g%% *}" 'BEGIN{exit !(g >= 0.0005)}' || { echo "S48L_STOP fold $f (blend gain ${g%% *} < +0.0005) $(date -Is)" >> $LOG; exit 0; }
done
echo "S48L_ADVANCE 3/3 folds >= +0.0005 $(date -Is)" >> $LOG
