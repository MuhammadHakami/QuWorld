#!/bin/bash
# doc 35 §29e: 3-D register / MLP twin writing into PCA [0, 32) on the frozen fusion trunk; ensembles; the registered pick.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; T=runs/nas/sophelio/trunks
until grep -q S29D_DONE $S; do sleep 60; done                  # one fusion driver at a time (each holds ~15 GB of cache)
while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 30 ]; do sleep 30; done
QW_GPU_GB=16 timeout 28800 runs/capped.sh s29e 32 .venv/bin/python paper/nas/sophelio_3d.py --k-lo 0 --k-hi 32 --tag sota3dc \
  >> runs/alloc/s29.log 2>&1 || echo "CRASH s29e train $(date -Is)" >> $S
until grep -qE "S29D_DONE" $S; do sleep 120; done
for tag in _reg3dc _mlp3dc; do
  pat=$([ $tag = _reg3dc ] && echo "sota3dc-qubits_seed*.pt" || echo "sota3dc-mlp_seed*.pt")
  [ -f runs/sophelio_sota/ens_select$tag.json ] || QW_GPU_GB=8 timeout 14400 runs/capped.sh s29e 16 .venv/bin/python paper/sophelio/ens_select.py \
     --glob "$T/trunk1_t*.pt" "$T/$pat" --tag $tag >> runs/alloc/s29.log 2>&1 || echo "CRASH s29e ens $tag $(date -Is)" >> $S
done
QW_GPU_GB=8 timeout 7200 runs/capped.sh s29e 16 .venv/bin/python paper/sophelio/score_test.py --tags _reg3dc _mlp3dc >> runs/alloc/s29.log 2>&1 \
  || echo "CRASH s29e test $(date -Is)" >> $S
best=$(python3 -c "
import json
v={t: max(x['S'] for x in json.load(open(f'runs/sophelio_sota/ens_select{t}.json'))['valid_full'].values()) for t in ('_reg3d','_reg3dc')}
print(max(v, key=v.get))")
echo "S29E pick ens_select$best.json (full-VALID) $(date -Is)" >> $S
QW_GPU_GB=8 timeout 3600 runs/capped.sh s29e 16 .venv/bin/python paper/sophelio/submit_public.py --entry ens_select$best.json \
  --out submission_3d_pick >> runs/alloc/s29.log 2>&1 || echo "CRASH s29e build $(date -Is)" >> $S
echo "S29E_DONE $(date -Is)" >> $S
