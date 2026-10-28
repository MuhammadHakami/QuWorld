#!/bin/bash
# doc 35 §23 TokaMark deviation: full recipe (ft_heads 5 + ft_full 15) for the 3-D register and its matched MLP, 16 h cap
# each, sequential; then their eval with per-window prediction dumps and the Figure 2 row.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; PY=.venv-tokamark/bin/python
R=runs/tokamark/results; mkdir -p $R/dumps
for v in q30-geo=3d q30-geo=3d-mlp; do
  while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done
  QW_GPU_GB=20 timeout 57600 runs/capped.sh s23tk 48 $PY paper/tokamark/tokamind_hybrid.py finetune --task task_1-1 --variant $v --seed 1337 \
     >> runs/s23_tokamark.log 2>&1 || { echo "CRASH s23b tokamark ft $v $(date -Is)" >> $S; continue; }
  TOKAMIND_DUMP_TRUTH=1 TOKAMIND_DUMP=$R/dumps/${v}_s1337.npz QW_GPU_GB=20 timeout 7200 runs/capped.sh s23tke 48 $PY paper/tokamark/tokamind_hybrid.py eval \
     --task task_1-1 --variant $v --seed 1337 >> runs/s23_tokamark.log 2>&1 || echo "CRASH s23b tokamark dump $v $(date -Is)" >> $S
  QW_GPU_GB=20 timeout 7200 runs/capped.sh s23tke 48 $PY paper/tokamark/tokamind_hybrid.py eval \
     --task task_1-1 --variant $v --seed 1337 >> runs/s23_tokamark.log 2>&1 || echo "CRASH s23b tokamark eval $v $(date -Is)" >> $S
done
.venv/bin/python paper/tokamark/dump_rows_s23.py >> runs/s23_tokamark.log 2>&1 || echo "CRASH s23b tokamark rows $(date -Is)" >> $S
echo "S23_TOKAMARK_DONE $(date -Is)" >> $S
