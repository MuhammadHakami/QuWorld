#!/bin/bash
# §48k (doc 35, registered before any result): psi recipe probes around t20 - fold 0 one-factor arms K1-K5 (target psi), comps 0-31 SSE vs Vpsi f0;
# advance iff <= 0.95 x Vpsi on f0, adopt iff <= 0.95 on >= 2/3 folds. One job at a time.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion; LOG=$L/s48k.log
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=20 QW_GPU_GB=24"
B='"target": "psi", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "q_noise": 0.0, "in_noise": 0.0, "mixer": 64, "reg": "tether3d"'
declare -A J=([K1]="{$B, \"bb_depth\": 4, \"tokens\": 12, \"crop\": 32, \"shots\": 128, \"huber\": 1.0}"
              [K2]="{$B, \"bb_depth\": 4, \"tokens\": 12, \"crop\": 64, \"shots\": 64, \"huber\": 1.0}"
              [K3]="{$B, \"bb_depth\": 6, \"tokens\": 12, \"crop\": 16, \"shots\": 256, \"huber\": 1.0}"
              [K4]="{$B, \"bb_depth\": 4, \"tokens\": 16, \"crop\": 16, \"shots\": 256, \"huber\": 1.0}"
              [K5]="{$B, \"bb_depth\": 4, \"tokens\": 12, \"crop\": 16, \"shots\": 256, \"huber\": 0.0}")
echo "[s48k] start $(date -Is)" >> $LOG
run() { [ -f $L/quota/$1_f$2.npz ] && return 0
  env $E runs/capped.sh s48k$1$2 36 .venv/bin/python paper/strat/fusion/quota_fusion.py cv $1 "${J[$1]}" $2 3 > $L/s48k_$1_f$2.log 2>&1
  echo "[s48k] $1 f$2 exit $? $(date -Is)" >> $LOG; }
ratio() { .venv/bin/python - "$1" "$2" <<'PY'
import sys, numpy as np
Q = "runs/strat/fusion/quota/"; a, k = sys.argv[1], sys.argv[2]
zr, za = np.load(Q + f"Vpsi_f{k}.npz"), np.load(Q + f"{a}_f{k}.npz"); assert (zr["rows"] == za["rows"]).all(); y = zr["y"].astype(np.float64)
e = lambda p: ((y - p) ** 2)[:, :32].sum()                                    # noqa: E731
print(f"{e(za['pred']) / e(zr['pred']):.4f}")
PY
}
ADV=()
for a in K1 K2 K3 K4 K5; do
  run $a 0; r=$(ratio $a 0 2>>$LOG); echo "S48K_F0 $a ratio comps0-31 SSE vs Vpsi ${r:-?} $(date -Is)" >> $LOG
  awk -v r="$r" 'BEGIN{exit !(r != "" && r <= 0.95)}' && ADV+=($a)
done
echo "[s48k] fold-0 advancing: ${ADV[*]:-none} $(date -Is)" >> $LOG
WIN=""
for a in "${ADV[@]}"; do n=1                                                  # fold 0 already passed
  for k in 1 2; do run $a $k; r=$(ratio $a $k 2>>$LOG); echo "S48K_F$k $a ratio ${r:-?} $(date -Is)" >> $LOG
    awk -v r="$r" 'BEGIN{exit !(r != "" && r <= 0.95)}' && n=$((n + 1)); done
  [ $n -ge 2 ] && WIN="$WIN $a($n/3)"
done
echo "S48K_DONE adopt:${WIN:- none} $(date -Is)" >> $LOG
