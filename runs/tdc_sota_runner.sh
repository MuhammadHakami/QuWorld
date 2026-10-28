#!/bin/bash
# TDC SoTA push (Sep 23): exact MapLight on CPU CatBoost (user-approved exception), then MapLight + frozen CheMeleon
# embeddings (the MapLight+GNN recipe with a stronger foundation model; embeddings on GPU), then MapLight/CheMeleon ensembles.
cd /home/mohhakami/Documents/paper/quworld; export QW_GPU_GB=4 PYTHONUNBUFFERED=1; S=runs/tdc_maplight/status; PY=.venv-tdc/bin/python
runs/capped.sh ml-cpu 16 $PY paper/tdc/run_maplight.py all >> runs/tdc_maplight/run.log 2>&1 && echo "MAPLIGHT_CPU_DONE $(date -Is)" >> $S || echo "CRASH maplight cpu $(date -Is)" >> $S
runs/capped.sh ml-chem 16 $PY paper/tdc/run_maplight.py all --features maplight+chemeleon >> runs/tdc_maplight/run_chem.log 2>&1 && echo "MAPLIGHT_CHEM_DONE $(date -Is)" >> $S || echo "CRASH maplight+chemeleon $(date -Is)" >> $S
$PY paper/tdc/ensemble.py > runs/tdc_ensemble/table.txt 2>&1; echo "TDC_ENSEMBLE_DONE $(date -Is)" >> $S
