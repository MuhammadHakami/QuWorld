#!/usr/bin/env bash
# Tokamak signals (TokaMark task 1-1 on MAST; Table 1 neck row): the released TokaMind foundation model with the
# register on its CLS latent, under TokaMind's own warm-start protocol (5 epochs heads + 15 epochs full), arms
# {vanilla, mlp30, mr30ryd} x seeds 1337-1339, then shot-level paired comparison of the 3-seed ensembles.
# Follows runs/tokamind_runner.sh; environment from runs/well_dl/venv_tokamark_install.sh + runs/tokamark/tokamind_patch.sh.
# Needs baseline/tokamark and baseline/tokamind (https://github.com/UKAEA-IBM-STFC-Fusion-FMs).
#   bash reproduce/tokamark.sh [env|data|train|eval|all]
source "$(dirname "$0")/common.sh"
TPY=.venv-tokamark/bin/python; D=paper/tokamark/tokamind_hybrid.py; L=runs/tokamark/tokamind; mkdir -p $L

if step env; then
  [ -d .venv-tokamark ] || uv venv -q .venv-tokamark --python 3.13
  uv pip install --python $TPY --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match "torch==2.13.0" torchvision
  uv pip install --python $TPY --index-strategy unsafe-best-match -e baseline/tokamark line-profiler torchinfo pyyaml
  bash runs/tokamark/tokamind_patch.sh                                                     # TokaMind editable install + py3.13 fix
fi

if step data; then   # 11,573 MAST shots as zarr (~35 GB) from the public FAIR-MAST bucket, and the TokaMind weights
  $TPY runs/tokamark/download_shots.py                                                      # -> data/tokamark/zarr/
  hf UKAEA-IBM-STFC/tokamind-base-v2 --local-dir data/tokamark/tokamind-base-v2
fi

if step train; then
  for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
    done_if $L/train_${v}_s${seed}.done && continue
    say "tokamind finetune $v seed $seed"
    $TPY $D finetune --task task_1-1 --variant $v --seed $seed --workers 8 > $L/finetune_${v}_s${seed}.log 2>&1 && touch $L/train_${v}_s${seed}.done
  done; done
fi

if step eval; then   # one TEST evaluation per model, then the paired 3-seed ensemble comparison
  for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
    done_if $L/eval_${v}_s${seed}.log || $TPY $D eval --task task_1-1 --variant $v --seed $seed --workers 8 > $L/eval_${v}_s${seed}.log 2>&1
  done; done
  py paper/tokamark/paired_ens.py                                                           # -> runs/tokamark/results/paired_ens3.json
fi
