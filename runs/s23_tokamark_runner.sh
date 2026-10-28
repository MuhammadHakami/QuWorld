#!/bin/bash
# doc 35 §23: TokaMark task 1-1 on the TokaMind trunk (their pipeline): warm-start fine-tune (their ft_heads 5 + ft_full 15
# epochs) for the 3-D register and its matched MLP, seed 1337, then their eval with per-window prediction dumps.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; PY=.venv-tokamark/bin/python
R=runs/tokamark/results; mkdir -p $R/dumps
until grep -q S23_1HP_DONE $S && grep -q S23_PSI_DONE $S; do sleep 60; done
TOKAMIND_DUMP_TRUTH=1 TOKAMIND_DUMP=$R/dumps/vanilla_s1337.npz QW_GPU_GB=20 timeout 7200 runs/capped.sh s23tkv 48 $PY paper/tokamark/tokamind_hybrid.py eval \
   --task task_1-1 --variant vanilla --seed 1337 >> runs/s23_tokamark.log 2>&1 || echo "CRASH s23 tokamark eval vanilla $(date -Is)" >> $S
for v in q30-geo=3d q30-geo=3d-mlp; do
  QW_GPU_GB=20 timeout 14400 runs/capped.sh s23tk 48 $PY paper/tokamark/tokamind_hybrid.py finetune --task task_1-1 --variant $v --seed 1337 \
     >> runs/s23_tokamark.log 2>&1 || echo "CRASH s23 tokamark ft $v $(date -Is)" >> $S
  TOKAMIND_DUMP_TRUTH=1 TOKAMIND_DUMP=$R/dumps/${v}_s1337.npz QW_GPU_GB=20 timeout 7200 runs/capped.sh s23tke 48 $PY paper/tokamark/tokamind_hybrid.py eval \
     --task task_1-1 --variant $v --seed 1337 >> runs/s23_tokamark.log 2>&1 || echo "CRASH s23 tokamark eval $v $(date -Is)" >> $S
done; echo "S23_TOKAMARK_DONE $(date -Is)" >> $S
