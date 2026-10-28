#!/bin/bash
# Fusion §42h promotion: iff PQ2_VB vs LS_ctrl_VB has delta > 0, CI lower bound > 0 and D_LCFS not worse (registered rule), train
# PQ2A s1-3 = psiFA recipe (PSI-F + all data + canonical currents) + qphys2.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42h_promote.log
until grep -q "^S42H_DONE" $L/s42h.log 2>/dev/null; do sleep 120; done
ok=$(python3 - $L/s42h.log <<'PY'
import re, sys
m = re.search(r"\[vtpp\] PQ2_VB vs LS_ctrl_VB: .*?delta ([-+0-9.]+) \[([-+0-9.]+), [-+0-9.]+\] \| .*?dlcfs ([-+0-9.]+)", open(sys.argv[1]).read())
print(int(bool(m) and float(m.group(1)) > 0 and float(m.group(2)) > 0 and float(m.group(3)) <= 0))
PY
)
echo "[pq2] rule met: $ok $(date -Is)" >> $LOG
[ "$ok" = 1 ] || { echo "S42H_PROMOTE_SKIP $(date -Is)" >> $LOG; exit 0; }
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "all_data": true, "canon": true, "qphys": 2'
for s in 2 1 3; do
  [ -f runs/sophelio_sota/PQ2A_s$s.pt ] || QW_GPU_GB=30 runs/capped.sh s42hp 40 .venv/bin/python paper/sophelio/sota_train.py --tag PQ2A_s$s --score-val 0 \
    --cfg "{$CFG, \"seed\": $s}" >> $LOG 2>&1 || echo "CRASH PQ2A s$s $(date -Is)" >> $LOG
done
echo "S42H_PROMOTE_DONE $(date -Is)" >> $LOG
