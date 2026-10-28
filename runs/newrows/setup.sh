#!/bin/bash
# New industry rows (Sep 21): traffic forecasting (BasicTS, METR-LA/PEMS-BAY) and TDC ADMET (drug discovery).
# Clones the SoTA repos, builds isolated envs, downloads data. Idempotent; log = runs/newrows/setup.log.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH"; L=runs/newrows/setup.log
log() { echo "$(date -Is) $*" >> $L; }
log "setup start"
[ -d baseline/BasicTS ] || git clone --depth 1 https://github.com/GestaltCogTeam/BasicTS baseline/BasicTS >> $L 2>&1 && log "BasicTS cloned"
[ -d baseline/TDC ] || git clone --depth 1 https://github.com/mims-harvard/TDC baseline/TDC >> $L 2>&1 && log "TDC cloned"
[ -d .venv-tdc ] || { uv venv --python 3.12 .venv-tdc >> $L 2>&1; uv pip install --python .venv-tdc/bin/python PyTDC torch rdkit scikit-learn pandas >> $L 2>&1 && log "venv-tdc ready"; }
# TDC ADMET group: download all 22 endpoints (scaffold split, 5 seeds) into data/tdc
.venv-tdc/bin/python - >> $L 2>&1 << 'PY'
from tdc.benchmark_group import admet_group
g = admet_group(path="data/tdc"); print("ADMET group endpoints:", g.dataset_names)
for name in g.dataset_names:
    b = g.get(name); print(name, "train_val", len(b["train_val"]), "test", len(b["test"]))
PY
log "tdc admet data done"
# METR-LA / PEMS-BAY raw data (BasicTS layout): try gdown from the BasicTS-published Google Drive folder
uv pip install --python .venv-tdc/bin/python gdown >> $L 2>&1
mkdir -p data/traffic; cd data/traffic
[ -f METR-LA/data.dat ] || .venv-tdc/bin/gdown --folder "https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lPKdBQ5KR" -O . >> ../../$L 2>&1 || log "gdown folder failed (needs manual download of BasicTS datasets)"
cd ../..; ls data/traffic >> $L 2>&1
log "setup end"
