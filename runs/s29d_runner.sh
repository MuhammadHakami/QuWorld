#!/bin/bash
# doc 35 §29b addendum: ensembles with the 3-D register / 3-D MLP pools, held-out TEST once, then the public-fold build.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; T=runs/nas/sophelio/trunks
until grep -q S29B_DONE $S; do sleep 120; done
for tag in _reg3d _mlp3d; do
  pat=$([ $tag = _reg3d ] && echo "sota3d-qubits_seed*.pt" || echo "sota3d-mlp_seed*.pt")
  [ -f runs/sophelio_sota/ens_select$tag.json ] || QW_GPU_GB=8 timeout 14400 runs/capped.sh s29d 16 .venv/bin/python paper/sophelio/ens_select.py \
     --glob "$T/trunk1_t*.pt" "$T/$pat" --tag $tag >> runs/alloc/s29.log 2>&1 || echo "CRASH s29d ens $tag $(date -Is)" >> $S
done
QW_GPU_GB=8 timeout 7200 runs/capped.sh s29d 16 .venv/bin/python paper/sophelio/score_test.py --tags _reg3d _mlp3d >> runs/alloc/s29.log 2>&1 \
  || echo "CRASH s29d test $(date -Is)" >> $S
QW_GPU_GB=8 timeout 3600 runs/capped.sh s29d 16 .venv/bin/python paper/sophelio/submit_public.py --entry ens_select_reg3d.json \
  --out submission_reg3d >> runs/alloc/s29.log 2>&1 || echo "CRASH s29d build $(date -Is)" >> $S
echo "S29D_DONE $(date -Is)" >> $S
