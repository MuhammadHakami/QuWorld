#!/usr/bin/env bash
# Fusion equilibrium (Sophelio DIII-D, Challenge 1; Table 1 neck row, score S, higher is better): the starter kit's own
# UNetLite and pipeline with the register on its vector latent; every arm tuned with Optuna (10 trials, objective = mean
# VALID over seeds 1337 + 1338), then scored once on TEST with seeds 1337-1339. Follows runs/hpo_none_ms_runner.sh and
# runs/hpo_ms_runner.sh. Step "best" rebuilds the strongest classical model ("our best", Appendix): full-data features,
# a trunk architecture search, seed replicas, a VALID-selected ensemble, one TEST score.
# Needs baseline/sophelio-starter (https://github.com/Sophelio/fusion-equilibrium-challenge-starter).
#   bash reproduce/fusion_equilibrium.sh [data|train|best|all]
source "$(dirname "$0")/common.sh"

if step data; then   # DIII-D training shots (93 GB of the 104 GB repository)
  hf Sophelio/fusion-equilibrium-challenge --repo-type dataset --include "data/diii_d_train/*" \
    --local-dir data/sophelio/fusion-equilibrium-challenge
  [ -d baseline/sophelio-starter ] || git clone https://github.com/Sophelio/fusion-equilibrium-challenge-starter baseline/sophelio-starter
fi

if step train; then  # arms: none (classical) | mlp (matched MLP) | q (30-atom register)
  for arm in none mlp q; do
    done_if runs/hpo/sophelio/${arm}_ms2_final.json && continue
    py paper/hpo/driver.py sophelio $arm search --trials 10 --search-seeds 1337 1338 --study ms2
    py paper/hpo/driver.py sophelio $arm final --study ms2
  done
fi

if step best; then   # VALID-only selection; TEST is read once by score_test.py
  py paper/sophelio/extract.py --split diii_d_train --workers 8                             # -> data/sophelio_proc/
  py paper/nas/sophelio_nas.py --stage trunk --trials 20 --study trunk1
  py paper/sophelio/replicas.py --study trunk1 --top 3 --seeds 1 2
  py paper/sophelio/ens_select.py --glob 'runs/nas/sophelio/trunks/trunk1_t*.pt' --n-select 100
  py paper/sophelio/score_test.py --tags "" _reg _mlp                                       # -> runs/sophelio_sota/test_scores.json
fi
