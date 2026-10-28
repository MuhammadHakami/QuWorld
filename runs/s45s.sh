#!/bin/bash
# §45s: canonical twins of the register heads (J1QHCS s1-3, TRAIN, concurrent as J1QH) and of the X-point heads (XP1CS_s1 TRAIN; XPACS s1-3 all data).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s45s.log; F=paper/strat/fusion
echo "[s45s] start $(date -Is)" >> $LOG
QH='{"block": "mlp", "width": 2048, "depth": 4, "drop": 0.1, "q_layers": 4, "q_in": 64, "link": "independent", "lr": 0.0012505271791067153, "alpha_mult": 1.0, "alpha_init": 0.1, "w_pi": 0.0}'
pids=(); for s in 1 2 3; do [ -f $L/head_boundary_v8_J1QHCS_s$s.pt ] || { env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 \
  QW_GPU_GB=20 runs/capped.sh qhcs$s 30 .venv/bin/python $F/hpo_qpikan.py --fixed "$QH" --tag J1QHCS --epochs 60 --seed $s > $L/J1QHCS_s$s.log 2>&1 & pids+=($!); }; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_CANON=2"
( [ -f $L/head_xp_XP1CS_s1.pt ] || env $E STRAT_OUT_TAG=_XP1CS QW_GPU_GB=20 runs/capped.sh xp1cs 30 .venv/bin/python $F/head_boundary.py 1 xp > $L/XP1CS_s1.log 2>&1
  for s in 1 2 3; do [ -f $L/head_xp_XPACS_s$s.pt ] || env $E STRAT_ALL=1 STRAT_OUT_TAG=_XPACS QW_GPU_GB=20 runs/capped.sh xpacs 30 .venv/bin/python $F/head_boundary.py $s xp >> $L/XPACS.log 2>&1; done ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || echo "CRASH s45s pid $p $(date -Is)" >> $LOG; done
echo "S45S_TRAINED $(date -Is)" >> $LOG
