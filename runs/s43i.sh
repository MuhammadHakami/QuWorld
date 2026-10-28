#!/bin/bash
# §43i (registered before any result): cause of §43h's fast-path loss (J1QPIpiF_s1 -0.000627 [-0.001229, -0.000112] vs J1QPIpi_s1).
# J1QPIpiN_s1 = J1QPIpi recipe on the layered fast path WITHOUT the inductor compile (QW_QFAST=1, QW_QCOMPILE unset), TRAIN only, seed 1.
# VT singles vs J1QPIpi_s1 (paired shot bootstrap). Rule: |J1QPIpiN - J1QPIpi| <= 0.0004 -> the compile is the cause (fast path ok
# without it); else the layered math itself (bf16 interplay) -> WL CNNs re-checked with QW_QFAST=0.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s43i.log
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
echo "[s43i] start $(date -Is)" >> $LOG
env STRAT_CUDA=0 QW_QFAST=1 QW_MAGFIX=X_magfix QW_GPU_GB=20 runs/capped.sh qJ1QPIpiN1 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py \
  --fixed "$PK" --tag J1QPIpiN --epochs 60 --seed 1 > $L/J1QPIpiN_s1.log 2>&1 && echo "J1QPIpiN_s1_TRAINED $(date -Is)" >> $LOG || { echo "CRASH J1QPIpiN $(date -Is)" >> $LOG; exit 1; }
H=head_boundary_v8
flock runs/strat/vt.lock .venv/bin/python paper/strat/fusion/vt_eval.py shape ${H}_J1QPIpi_s1 ${H}_J1QPIpiN_s1 ${H}_J1QPIpiF_s1 >> $LOG 2>&1 || { echo "CRASH vt $(date -Is)" >> $LOG; exit 1; }
echo "S43I_DONE $(date -Is)" >> $LOG
