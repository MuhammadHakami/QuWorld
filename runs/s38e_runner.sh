#!/bin/bash
# §38e head capacity screen (one seed each, fast mode)
cd /home/mohhakami/Documents/paper/quworld
H="env STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_SCAL_W=9.142857 STRAT_FAST=1"
run() { tag=$1; shift; flock runs/strat/gpu.lock $H "$@" STRAT_OUT_TAG=_$tag QW_GPU_GB=24 runs/capped.sh s38e 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 > runs/strat/fusion/s38e_$tag.log 2>&1; }
run H0 STRAT_EPOCHS=60
run H1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=3
run H2 STRAT_EPOCHS=120
.venv/bin/python paper/strat/fusion/head_screen.py head_boundary_v8_s1 head_boundary_v8_H0_s1 head_boundary_v8_H1_s1 head_boundary_v8_H2_s1 > runs/strat/fusion/s38e_screen.log 2>&1
echo S38E_DONE >> runs/strat/fusion/s38e_screen.log
