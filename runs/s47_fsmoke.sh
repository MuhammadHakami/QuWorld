#!/bin/bash
# §47 integration smoke: final mode (1 epoch, all TRAIN) -> checkpoint; gpu_build loader on the VALID-sample shots must reproduce its predictions.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47_fsmoke.log
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=30"
env $E runs/capped.sh s47fs 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final Vsmoke '{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}' 1 1 >> $LOG 2>&1
env $E .venv/bin/python - >> $LOG 2>&1 <<'PY'
import sys, numpy as np, torch
sys.path[:0] = ["paper/strat/fusion", "paper/sophelio", "paper"]
import fx, sota, gpu_build as GB
sota.DEV = torch.device("cuda"); d = sota.Data()
z = np.load(fx.OUT / "head_boundary_v8_Vsmoke_s1.npz"); va = z["rows_valid"]; shots = np.unique(d.shot[va])
rows = np.nonzero(np.isin(d.shot, shots))[0]
p = GB.heads_gpu(d, rows, "head_boundary_v8_Vsmoke_s1"); pos = {r: i for i, r in enumerate(rows)}
q = p[[pos[r] for r in va]]; diff = np.abs(q - z["pred_valid"]).max(0)
print("[fsmoke] loader vs trainer max |diff| scalars", np.round(diff[:7], 5), "rays max", float(diff[7:].max()), "rows", len(va), "whole-shot rows", len(rows))
PY
echo "[fsmoke] exit $? $(date -Is)" >> $LOG
