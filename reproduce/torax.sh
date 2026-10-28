#!/usr/bin/env bash
# Tokamak transport (TORAX-Bench, Table 1 neck row): generate the benchmark with TORAX, then tune every arm with Optuna
# on TRAIN/VALID (16 trials, objective = mean VALID over seeds 1337 + 1338) and score the selected configuration once on
# TEST with seeds 1337-1339. Follows runs/hpo_none_ms_runner.sh and runs/hpo_ms_runner.sh (study "ms2").
# The generator needs its own environment: TORAX v1.4.3 (JAX, CPU, float64); point TORAX_PY at its python.
#   bash reproduce/torax.sh [data|train|all]
source "$(dirname "$0")/common.sh"
TORAX_PY=${TORAX_PY:-.venv-torax/bin/python}

if step data; then   # 1,000 discharges x 121 frames, 12-D actuator space, seed 20260816, frozen 800/100/100 splits
  done_if data/torax_bench_v1/manifest.json || $TORAX_PY paper/torax_bench/generate_v1.py 1000 data/torax_bench_v1
  py paper/torax_bench/baselines.py data/torax_bench_v1 paper/torax_bench/splits.json         # persistence, climatology
fi

if step train; then  # arms: none (classical) | mlp (matched MLP) | q (30-atom register)
  for arm in none mlp q; do
    done_if runs/hpo/torax/${arm}_ms2_final.json && continue
    py paper/hpo/driver.py torax $arm search --trials 16 --search-seeds 1337 1338 --study ms2
    py paper/hpo/driver.py torax $arm final --study ms2                                      # TEST once, seeds 1337-1339
  done
fi
