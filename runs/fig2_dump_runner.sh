#!/bin/bash
# Fig. 2 (user Oct 3): routed co-processor outputs on one TEST window (the middle one) for the Table 1 rows:
# classical (gate + linear head), MLP 30, register 30, register 50; seed 1, Table 1 flags; results in a separate dir.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 QW_OUT=runs/alloc/fig2_dump; D=runs/alloc/fig2_dump
for row in shear_flow__gate_vanilla_s1337 rayleigh_benard__gatestfull_vanilla_s1337 turbulent_radiative_layer_2D__gate_vanilla_s1337 MHD_64__gate_vanilla_s1337; do
  N=$(.venv/bin/python -c "import torch; print(torch.load('runs/alloc/adrc_cache/$row/test.pt', weights_only=False, mmap=True)['a'].shape[0])")
  W=$((N / 2)); echo "$row N=$N window=$W" >> $D/runner.log
  for spec in "classical:cls_only:" "mlp:adrc_mlp:--route_sup --dyn_tether" "qubits:adrc_ryd:--route_sup --dyn_tether" "qubits50:adrc_ryd:--route_sup --dyn_tether --atoms 50"; do
    IFS=: read tag arm flags <<< "$spec"
    [ -f $D/${row}_$tag.npz ] && continue
    QW_DUMP_WINDOW=$D/${row}_$tag.npz:$W QW_GPU_GB=16 runs/capped.sh fig2 30 .venv/bin/python paper/alloc/adrc.py $row --arm $arm --seed 1 --gate $flags \
      >> $D/runner.log 2>&1 || echo "CRASH $row $tag" >> $D/runner.log
  done
done
echo FIG2_DUMP_DONE >> $D/runner.log
