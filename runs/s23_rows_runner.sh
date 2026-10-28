#!/bin/bash
# Figure 2 rows (fig_outputs_all.py) for the Well benchmarks: as each doc 35 §23 chain finishes (classical, matched 3-D MLP,
# 3-D tethered register), dump its fixed TEST sample (MHD: validation trajectory 0) to paper/fig_cache/rows_s23/;
# runs/paper_refresh.sh then regenerates the figure and rebuilds the paper. No training, no new TEST scoring of models.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; R=paper/fig_cache/rows_s23
mem() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt ${1:-30} ]; do sleep 60; done; }
dump() { local marker=$1 gb=$2 cap=$3; shift 3
  until grep -q "$marker" $S; do sleep 120; done; mem $cap
  QW_GPU_GB=$gb timeout 7200 runs/capped.sh rows $cap .venv/bin/python "$@" >> runs/alloc/rows_s23.log 2>&1 \
    || echo "CRASH s23 rows $* $(date -Is)" >> $S; }
[ -f $R/shear_flow.npz ] || { until grep -q S23_WELL_shear_flow_DONE $S; do sleep 120; done
  dump S23_WELL_rayleigh_benard_DONE 12 32 paper/alloc/dump_rows_well_s23.py shear_flow rayleigh_benard; }
[ -f $R/turbulent_radiative_layer_2D.npz ] || dump S23_TRL2D_DONE 12 32 paper/alloc/dump_rows_well_s23.py turbulent_radiative_layer_2D
[ -f $R/MHD_64.npz ] || dump S23_MHD_DONE 28 60 paper/alloc/dump_rows_mhd_s23.py
echo "S23_ROWS_DONE $(date -Is)" >> $S
