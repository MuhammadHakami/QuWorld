#!/bin/bash
# §47f: V trained 60 epochs on all TRAIN (seeds 1-3, two at a time), then the head-level VALID-sample read vs V.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47f.log
V='{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}'
echo "[s47f] start $(date -Is)" >> $LOG
tr() { [ -f $L/head_boundary_v8_V60_s$1.pt ] && return 0
  env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=30 \
    runs/capped.sh s47fV$1 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final V60 "$V" $1 60 > $L/s47f_V60_s$1.log 2>&1
  echo "[s47f] V60 s$1 exit $? $(date -Is)" >> $LOG; }
tr 1 & tr 2 & wait; tr 3
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import numpy as np
L = np.load("runs/strat/fusion/boundary_labels_v8.npz"); y = L["y_valid"].astype(np.float64)
def err(stem):
    p = np.mean([np.load(f"runs/strat/fusion/head_boundary_v8_{stem}_s{s}.npz")["pred_valid"] for s in (1, 2, 3)], 0); out = []
    for k in range(71):
        m = np.isfinite(y[:, k]); out.append(np.mean((y[m, k] - p[m, k]) ** 2) / np.var(y[m, k]))
    return np.array(out)
a, b = err("V"), err("V60")
print(f"[s47f] VALID sample 1-R2 mean-7 V {a[:7].mean():.5f} V60 {b[:7].mean():.5f} ratio {b[:7].mean() / a[:7].mean():.3f} | rays V {a[7:].mean():.5f} V60 {b[7:].mean():.5f} ratio {b[7:].mean() / a[7:].mean():.3f}")
PY
echo "S47F_DONE $(date -Is)" >> $LOG
