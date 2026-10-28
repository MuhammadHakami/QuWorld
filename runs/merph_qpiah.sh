#!/bin/bash
# Fusion §41u (merphistophline): register heads upgraded to the J1AH input set - PI-KAN + simulated 30-atom Rydberg layer (J1QPIpi
# recipe) on all labelled shots with corrected magnetics + fixed-coordinate Thomson + full-rate history (J1QPIpiAH), seeds 1-2,
# after J1AH s8. Replaces the TRAIN-only, no-thz J1QPIpi s1-2 in the final ensemble (register share 2/14).
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/fusion
until grep -q J1AH78_MERPH_DONE $L/J1AH_s7.log 2>/dev/null; do sleep 60; done
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
for s in 1 2; do
  [ -f $L/head_boundary_v8_J1QPIpiAH_s$s.pt ] && continue
  env STRAT_CUDA=0 STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 QW_MAGFIX=X_magfix python3 paper/strat/fusion/hpo_qpikan.py --fixed "$PK" --tag J1QPIpiAH \
    --epochs 60 --seed $s > $L/J1QPIpiAH_s$s.log 2>&1 || echo "CRASH J1QPIpiAH s$s $(date -Is)" >> $L/J1QPIpiAH_s$s.log
done
echo "QPIAH_MERPH_DONE $(date -Is)" >> $L/J1QPIpiAH_s1.log
