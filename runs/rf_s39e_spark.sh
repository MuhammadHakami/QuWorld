#!/bin/bash
# doc 35 §39e on the Spark: sampling decodes for seeds 5 and 4 (mephiles does seeds 1-3; checkpoints copied from mephiles).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/routing/s39e_spark.log
flock runs/strat/gpu.lock env QW_GPU_GB=30 runs/capped.sh s39e 30 .venv-vrp/bin/python paper/strat/routing/test_decode.py --budget 30000 \
  --jobs route30@bf16_mephiles@5:s8x4t05,vanilla@bf16_mephiles@5:s8x4t05,route30@bf16_mephiles@4:s8x4t05,vanilla@bf16_mephiles@4:s8x4t05 \
  >> $L 2>&1 || echo "CRASH $(date -Is)" >> $L
echo "S39E_SPARK_DONE $(date -Is)" >> $L
