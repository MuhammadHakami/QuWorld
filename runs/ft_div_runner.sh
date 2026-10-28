#!/bin/bash
# Descriptive E-ft diversity (doc 37): residual correlations on stratified VALID, after each row's ensemble exists.
cd /home/mohhakami/Documents/paper/quworld; S=runs/alloc/status; L=runs/alloc/ft.log
gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
for d in rayleigh_benard MHD_64 turbulent_radiative_layer_2D shear_flow; do
  while [ ! -f runs/alloc/ft/${d}_Eft.json ] && pgrep -x -f "/bin/bash runs/ft_ens_runner.sh" > /dev/null; do sleep 120; done
  gate; runs/capped.sh div-$d 24 .venv/bin/python paper/alloc/ft_ensemble.py $d --diversity >> $L 2>&1 || echo "CRASH ft-div $d $(date -Is)" >> $S
done
echo "FT_DIV_DONE $(date -Is)" >> $S
