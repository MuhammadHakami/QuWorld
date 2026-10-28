#!/bin/bash
# §48c smoke 2 (1 epoch, fold 0 of the §47 3-fold split): every new option at once - tethered x, y, z register + physical noise, input
# noise, Huber, 128-frame crops, depth 3 / width 160 / mixer 80 / 12 tokens / 6 register layers.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion
until grep -q '^S48_SMOKE_DONE' $L/s48_smoke.log; do sleep 30; done
J='{"backbone":"vhope","seq":true,"crop":128,"shots":32,"bb_width":160,"bb_depth":3,"mixer":80,"tokens":12,"q_layers":6,"q_in":96,"reg":"tether3d","q_noise":0.05,"in_noise":0.1,"huber":0.3,"alpha_init":0.2}'
env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QF_EPOCHS=1 QW_GPU_GB=24 \
  runs/capped.sh s48s2 36 .venv/bin/python paper/strat/fusion/quota_fusion.py cv smokeNz "$J" 0 3
echo "S48_SMOKE2_DONE $? $(date -Is)"
