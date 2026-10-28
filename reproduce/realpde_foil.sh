#!/usr/bin/env bash
# Real flow, airfoil (RealPDEBench foil; Table 1 bottom row): the released U-Net (fine-tuned stage) as the frozen
# classical part, plus a low-mode corrector with arms {none, lin, mlp30, mr30ryd} x seeds 1-3. Follows
# runs/realpde_arms_runner.sh. Needs baseline/RealPDEBench (https://github.com/AI4Science-WestlakeU/RealPDEBench).
#   bash reproduce/realpde_foil.sh [data|anchor|train|all]
source "$(dirname "$0")/common.sh"

if step data; then   # the airfoil subset only (the full repository is about 2 TB)
  hf AI4Science-WestlakeU/RealPDEBench --repo-type dataset --include "foil/*" --local-dir data/realpdebench
  hf AI4Science-WestlakeU/RealPDEBench-models --include "foil/*" --local-dir data/realpdebench_models
fi

if step anchor; then # RealPDEBench's own protocol on the released trunks (-> runs/realpde/trunk_<model>_finetune.json)
  py paper/realpde/eval_trunk.py --models unet fno cno --stage finetune
fi

if step train; then  # cache the trunk's TRAIN predictions once, then train the correctors (TEST once per arm and seed)
  done_if data/realpde_cache/pred_unet_finetune_train.npy || py paper/realpde/predict_split.py --model unet --split train
  py paper/realpde/arms.py --trunk unet --arms none lin mlp30 mr30ryd --seeds 1 2 3
fi
