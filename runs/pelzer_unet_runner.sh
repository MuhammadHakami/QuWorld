#!/bin/bash
# Pelzer 1HP — 1HP_NN U-Net track on the 1000-run set (GPU). Phase 1: LR sweep (s1337, 3000 steps) -> best val MAE;
# Phase 2: pretrain s1338/s1339 at the best LR; Phase 3: arms {vanilla, mlp30, mr30ryd} x 3 seeds, 1500 steps, lr 1e-4,
# each from the same-seed pretrained U-Net. Status: runs/pelzer_unet/status; sentinel PELZER_UNET_DONE.
export PATH="$HOME/.local/bin:$PATH"; cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1
O=runs/pelzer_unet; S=$O/status; mkdir -p $O; echo "runner start $(date -Is)" >> "$S"
PRE_STEPS=3000; ARM_STEPS=1500; ARM_LR=1e-4
for lr in 1e-3 3e-4 1e-4; do
  [ -f $O/pretrain_lr${lr}_s1337.json ] && continue
  echo "pretrain lr$lr s1337 $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py pretrain $lr 1337 $PRE_STEPS > $O/pretrain_lr${lr}_s1337.log 2>&1 || echo "CRASH pretrain lr$lr $(date -Is)" >> "$S"
done
BEST=$(python3 - << 'PY'
import json, glob
r = [(json.load(open(f))["best_val_mae_c"], f) for f in glob.glob("runs/pelzer_unet/pretrain_lr*_s1337.json")]
r.sort(); print(json.load(open(r[0][1]))["lr"])
PY
)
LRTAG=$(python3 -c "print(f'{float(\"$BEST\"):g}')"); echo "best pretrain lr $BEST (tag $LRTAG) $(date -Is)" >> "$S"; echo "$BEST" > $O/best_lr
for seed in 1338 1339; do
  [ -f $O/pretrain_lr${LRTAG}_s${seed}.json ] && continue
  echo "pretrain lr$BEST s$seed $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py pretrain $BEST $seed $PRE_STEPS > $O/pretrain_lr${LRTAG}_s${seed}.log 2>&1 || echo "CRASH pretrain s$seed $(date -Is)" >> "$S"
done
for seed in 1337 1338 1339; do for v in vanilla mlp30 mr30ryd; do
  [ -f $O/${v}_s${seed}.json ] && continue
  echo "arm $v s$seed $(date -Is)" >> "$S"
  uv run --no-sync python paper/geothermal/train_pelzer_unet.py arm $v $seed $ARM_STEPS $ARM_LR $O/pretrain_lr${LRTAG}_s${seed}.pt > $O/${v}_s${seed}.log 2>&1 || echo "CRASH arm $v s$seed $(date -Is)" >> "$S"
done; done
echo "PELZER_UNET_DONE $(date -Is)" >> "$S"
