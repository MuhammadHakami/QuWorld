#!/bin/bash
# Fusion §41q F1, hybrid part (competition rule of this project: classical + 30-atom register): J1QPIpi register heads (§38ad recipe:
# J1 architecture + engaged register, physics weight 0.1, 60 epochs) retrained on ALL 7,041 shots with the fixed-coordinate Thomson
# inputs (STRAT_ALL=1 STRAT_THZ=1), seeds 1 / 2, after the classical F1 components (s41q_final.sh). ~4.2 h per seed on the Spark.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s41q_final2.log
until grep -q S41Q_FINAL_DONE $L/s41q_final.log 2>/dev/null; do sleep 60; done
P='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
for s in 1 2; do
  [ -f $L/head_boundary_v8_J1QPIpiA_s$s.pt ] && continue
  flock runs/strat/gpu.lock env STRAT_ALL=1 STRAT_THZ=1 QW_GPU_GB=60 runs/capped.sh s41qf2 60 .venv/bin/python paper/strat/fusion/hpo_qpikan.py \
    --fixed "$P" --tag J1QPIpiA --epochs 60 --seed $s >> $LOG 2>&1 || echo "CRASH J1QPIpiA s$s $(date -Is)" >> $LOG
done
echo "S41Q_FINAL2_DONE $(date -Is)" >> $LOG
