#!/bin/bash
# §48m production seed, started speculatively while the §48m VT reads run (used only if S48M_ADOPT): PSIV60A s3 (all data, STRAT_ALL=1, 60 epochs),
# the PSIV60 recipe - the F21K addition. If §48m rejects, the checkpoint is unused.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=$PWD/runs/strat/fusion
J='{"target": "psi", "backbone": "vhope", "neck": "gate", "win": 8, "stride": 1, "bb_width": 128, "bb_depth": 4, "width": 2048, "depth": 4, "drop": 0.0, "q_layers": 4, "q_in": 32, "link": "independent", "lr": 0.000977105771213329, "alpha_init": 0.11830330369409621, "seq": true, "crop": 16, "shots": 256, "tokens": 12, "q_noise": 0.0, "in_noise": 0.0, "huber": 1.0, "mixer": 64, "reg": "tether3d"}'
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2"
[ -f $L/head_boundary_v8_PSIV60A_s3.pt ] || env $E QW_GPU_GB=30 runs/capped.sh s48mA3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final PSIV60A "$J" 3 60 \
  > $L/s48m_PSIV60A_s3.log 2>&1
echo "[s48mA] PSIV60A s3 exit $? $(ls -la $L/head_boundary_v8_PSIV60A_s3.pt 2>&1 | awk '{print $5}') $(date -Is)" >> $L/s48m.log
