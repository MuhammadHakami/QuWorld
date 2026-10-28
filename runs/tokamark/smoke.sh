#!/bin/bash
# TokaMark smoke: wait for the first 8 shots of each official split, then train the quantum arm for 60 steps on them (GPU) and evaluate.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/tokamark/smoke.log
need() { .venv-tokamark/bin/python - << 'PY'
import os, sys
from tokamark.data_split import get_train_test_val_shots
from tokamark.tools.path import RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
tr, te, va = get_train_test_val_shots(max_index=8, shuffle=True, data_splits_file_path=RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE)
miss = [s for s in tr + te + va if not os.path.exists(f"data/tokamark/zarr/{s}.zarr/.complete")]
sys.exit(1 if miss else 0)
PY
}
while ! need; do sleep 60; done
echo "$(date -Is) smoke: the 24 shots of their shuffled 8-per-split selection present; training mr30ryd 60 steps on 8 shots" >> $L
.venv-tokamark/bin/python paper/tokamark/train_hybrid.py --task task_1-1 --config /home/mohhakami/Documents/paper/quworld/paper/tokamark/config_smoke.yaml --seed 23 --variant mr30ryd --subset 8 --validate_every 20 >> $L 2>&1; echo "$(date -Is) smoke train exit $?" >> $L
.venv-tokamark/bin/python paper/tokamark/eval_hybrid.py --task task_1-1 --config /home/mohhakami/Documents/paper/quworld/paper/tokamark/config_smoke.yaml --seed 23 --variant mr30ryd --subset 8 >> $L 2>&1; echo "$(date -Is) smoke eval exit $? SMOKE_DONE" >> $L
