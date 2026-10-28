#!/bin/bash
# MapLight on GPU CatBoost (254 borders, MVS 0.8, Plain; Exact leaves crash on GPU): caco2 check vs published 0.276,
# then all 22 endpoints if caco2 MAE <= 0.285. Resumable (finished endpoints skipped).
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=6; S=runs/tdc_maplight/status
rm -f runs/tdc_maplight/caco2_wang.json
runs/capped.sh maplight 12 .venv-tdc/bin/python paper/tdc/run_maplight.py caco2_wang > runs/tdc_maplight/caco2_recheck.log 2>&1
m=$(python3 -c "import json; print(json.load(open('runs/tdc_maplight/caco2_wang.json'))['test_mean'])" 2>/dev/null)
echo "caco2 recheck MAE $m $(date -Is)" >> $S
python3 -c "import sys; sys.exit(0 if float('$m' or 9) <= 0.285 else 1)" || { echo "MAPLIGHT_NOT_REPRODUCED (caco2 $m > 0.285) $(date -Is)" >> $S; exit 0; }
runs/capped.sh maplight 12 .venv-tdc/bin/python paper/tdc/run_maplight.py all >> runs/tdc_maplight/run.log 2>&1 && echo "MAPLIGHT_DONE $(date -Is)" >> $S || echo "CRASH maplight $(date -Is)" >> $S
