#!/bin/bash
# doc 35 §38ae: HJl9 with the hybrid-augmented head ensemble (6 x J1 + J1QPI + J1QPIpi) -> VALID sample gate (official);
# then two more J1QPIpi seeds (2, 3) for a stronger classical + quantum ensemble.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
H="$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done)head_boundary_v8_J1QPI_s1+head_boundary_v8_J1QPIpi_s1"
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag HJQ8 --psi psiF --qb qh50 --heads "$H" \
  --kw "$KW" --axis head >> $L/s38ae2.log 2>&1
echo "S38AE2_SAMPLE_DONE $(date -Is)" >> $L/s38ae2.log
P='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
for s in 2 3; do
  flock runs/strat/gpu.lock env QW_GPU_GB=60 runs/capped.sh s38ae 60 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --fixed "$P" --tag J1QPIpi \
    --epochs 60 --seed $s >> $L/s38ae2.log 2>&1
done
echo "S38AE2_SEEDS_DONE $(date -Is)" >> $L/s38ae2.log
