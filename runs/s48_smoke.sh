#!/bin/bash
# §48c smoke: 1 epoch, separate directory (quota_smoke): control folds 0-3, then one search trial (the enqueued V reference).
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH HPO_DIR=quota_smoke HPO_EPOCHS=1
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=20"
env $E runs/capped.sh s48s 30 .venv/bin/python paper/strat/fusion/hpo_vq.py ctrl 0,1,2,3 && \
env $E HPO_WORKER=0 runs/capped.sh s48s 30 .venv/bin/python paper/strat/fusion/hpo_vq.py search 1 && \
.venv/bin/python paper/strat/fusion/hpo_vq.py report
echo "S48_SMOKE_DONE $? $(date -Is)"
