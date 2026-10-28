#!/bin/bash
# §42t step 2 (registered before any head result): X-point labels -> X-point head XP1_s1 (J1RL recipe, mode xp, TRAIN only) ->
# head-vs-truth X-point error on VT -> VT-wide chain arms VB / VBxh10 / VBxh0 (PQmix~g psi, TRAIN-only J1 x6 / QH x6), paired vs VB.
# Rule: adopt xh (XPA all-data heads + F-build) iff the better xh arm's S delta > 0 with CI > 0 and D_LCFS not worse.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion; LOG=$L/s42t_head.log
until grep -q "^S42T_DONE" $L/xpo.log 2>/dev/null; do sleep 30; done
[ -f $L/xp_labels.npz ] || QW_GPU_GB=16 runs/capped.sh xpl 30 .venv/bin/python paper/strat/fusion/xp_labels.py >> $LOG 2>&1 || { echo "CRASH xpl $(date -Is)" >> $LOG; exit 1; }
J="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
[ -f $L/head_xp_XP1_s1.pt ] || env $J STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_OUT_TAG=_XP1 QW_GPU_GB=30 \
  runs/capped.sh hXP1 40 .venv/bin/python paper/strat/fusion/head_boundary.py 1 xp >> $LOG 2>&1 || { echo "CRASH XP1 $(date -Is)" >> $LOG; exit 1; }
echo "XP1_TRAINED $(date -Is)" >> $LOG
cd paper/strat/fusion && ../../../.venv/bin/python - >> ../../../$LOG 2>&1 <<'PY'
import sys, numpy as np, torch, fx
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota; sota.DEV = torch.device("cuda"); import gpu_build as GB
d = sota.Data(); y = np.load(fx.OUT / "xp_labels.npz")["y"]; vt = np.nonzero((d.split != 0) & np.isfinite(y).any(1))[0]
p = GB.heads_gpu(d, vt, "head_xp_XP1_s1"); lo = np.isfinite(y[vt, 0])
e = np.where(lo, np.hypot(p[:, 0] - y[vt, 0], p[:, 1] - y[vt, 1]), np.hypot(p[:, 2] - y[vt, 2], p[:, 3] - y[vt, 3]))
print(f"[xph] VT diverted frames {len(vt)}: head |dX| median {np.median(e) * 1e3:.1f} mm, p75 {np.quantile(e, 0.75) * 1e3:.1f}, p90 {np.quantile(e, 0.9) * 1e3:.1f}"
      f" | dZ median {np.median(np.abs(np.where(lo, p[:, 1] - y[vt, 1], p[:, 3] - y[vt, 3]))) * 1e3:.1f} mm", flush=True)
PY
cd ../../..
Jh=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
X='"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}'
KWS="{\"VBc\":{\"_steps\":[\"shift\",\"bw\",\"kw\",\"xp\",\"jgn\",\"li\"],$X},\"VBxh10\":{\"_steps\":[\"shift\",\"bw\",\"kw\",\"xp\",\"jgn\",\"xh\",\"li\"],$X,\"xh\":{\"gate\":0.01}},\"VBxh0\":{\"_steps\":[\"shift\",\"bw\",\"kw\",\"xp\",\"jgn\",\"xh\",\"li\"],$X,\"xh\":{\"gate\":0.0}}}"
env QW_GPU_GB=20 runs/capped.sh s42tsm 30 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --vt-n 2 --tag XHs \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1 --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH smoke XHs $(date -Is)" >> $LOG; exit 1; }
echo "XH_SMOKE_OK $(date -Is)" >> $LOG
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s42tvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag XH \
  --psi psiF --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "$Q" --heads "$Jh" --xpheads head_xp_XP1_s1 --kws "$KWS" --axis head >> $LOG 2>&1 || { echo "CRASH vt XH $(date -Is)" >> $LOG; exit 1; }
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py "PQmix~g_VBc" "PQmix~g_VBxh10" "PQmix~g_VBxh0" >> ../../../$LOG 2>&1
echo "S42T_HEAD_DONE $(date -Is)" >> ../../../$LOG
