#!/bin/bash
# doc 35 §38ae (re-gate): after J1QPIpi seeds 2-3 - head screen and VALID sample (official) of HJl9 with 6 x J1 + J1QPI + 3 x J1QPIpi.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while ! grep -q "S38AE2_SEEDS_DONE" $L/s38ae2.log 2>/dev/null; do sleep 60; done
J="$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done)"; Q="head_boundary_v8_J1QPI_s1+head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2+head_boundary_v8_J1QPIpi_s3"
.venv/bin/python paper/strat/fusion/head_screen.py head_boundary_v8_J1QPIpi_s2 head_boundary_v8_J1QPIpi_s3 "${J%+}" "$J$Q" >> $L/s38ae3.log 2>&1
KW='{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}}'
flock runs/strat/gpu.lock .venv/bin/python paper/strat/fusion/gpu_build.py --split sample --tag HJQ10 --psi psiF --qb qh50 --heads "$J$Q" \
  --kw "$KW" --axis head >> $L/s38ae3.log 2>&1
echo "S38AE3_DONE $(date -Is)" >> $L/s38ae3.log
