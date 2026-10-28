#!/bin/bash
# Registered E-ft (doc 37): VALID selection per row, then one full TEST of the selection. shear waits for its MLP-neck cell.
cd /home/mohhakami/Documents/paper/quworld; S=runs/alloc/status; L=runs/alloc/ft.log
gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 60; done; }
run() { gate; runs/capped.sh ens-$1 24 .venv/bin/python paper/alloc/ft_ensemble.py $1 $2 >> $L 2>&1 || echo "CRASH ft-ens $1 $2 $(date -Is)" >> $S; }
for d in turbulent_radiative_layer_2D rayleigh_benard MHD_64; do run $d; run $d --full-test; done
while [ ! -f runs/alloc/ft/shear_flow_mlp30_vrmse_gate_s1_pre.json ] && pgrep -x -f "/bin/bash runs/ft_requeue_shear_mlp.sh" > /dev/null; do sleep 300; done
run shear_flow; run shear_flow --full-test
run turbulent_radiative_layer_2D; run turbulent_radiative_layer_2D --full-test
echo "FT_ENS_DONE $(date -Is)" >> $S
