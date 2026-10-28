#!/bin/bash
# §47f (continued): seed 3 now (seeds 1-2 already running), then the head-level VALID-sample read; if V60's error is >= 8% below V's
# (mean-7 ratio <= 0.92), the §45y-protocol VT read f17eq vs f17V60 (V60 s1-3 at share 0.25) with the §47c adoption rule.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47f.log
V='{"backbone":"vhope","seq":true,"crop":32,"shots":128,"bb_width":128,"bb_depth":2,"tokens":8}'
echo "[s47f] seed 3 started early, runner 2 $(date -Is)" >> $LOG
[ -f $L/head_boundary_v8_V60_s3.pt ] || env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=30 \
  runs/capped.sh s47fV3 40 .venv/bin/python paper/strat/fusion/quota_fusion.py final V60 "$V" 3 60 > $L/s47f_V60_s3.log 2>&1
echo "[s47f] V60 s3 exit $? $(date -Is)" >> $LOG
until [ -f $L/head_boundary_v8_V60_s1.pt ] && [ -f $L/head_boundary_v8_V60_s2.pt ]; do sleep 60; done; echo "[s47f] V60 s1 s2 done $(date -Is)" >> $LOG
R=$(.venv/bin/python - <<'PY'
import numpy as np
L = np.load("runs/strat/fusion/boundary_labels_v8.npz"); y = L["y_valid"].astype(np.float64)
def err(stem):
    p = np.mean([np.load(f"runs/strat/fusion/head_boundary_v8_{stem}_s{s}.npz")["pred_valid"] for s in (1, 2, 3)], 0); out = []
    for k in range(71):
        m = np.isfinite(y[:, k]); out.append(np.mean((y[m, k] - p[m, k]) ** 2) / np.var(y[m, k]))
    return np.array(out)
a, b = err("V"), err("V60"); r = b[:7].mean() / a[:7].mean()
print(f"{r:.4f} VALID sample 1-R2 mean-7 V {a[:7].mean():.5f} V60 {b[:7].mean():.5f} | rays V {a[7:].mean():.5f} V60 {b[7:].mean():.5f} ratio {b[7:].mean() / a[7:].mean():.3f}")
PY
); echo "[s47f] VALID ratio $R" >> $LOG
if awk -v r="${R%% *}" 'BEGIN{exit !(r <= 0.92)}'; then
  SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2)
  C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
  QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
  N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*4.5+head_boundary_v8_V60_s2*4.5+head_boundary_v8_V60_s3*4.5"
  H="route:$N+$QPI+$QHCS+$VV|$R_+$QPI+$QHCS+$VV"
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s47f 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S47Ff17V60 --psi psiF \
    --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
    --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"f17V60\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
    || { echo "S47F_FAIL vt $(date -Is)" >> $LOG; exit 1; }
  .venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17eq" "PQmix~g_f17V60" >> $LOG 2>&1
else echo "[s47f] V60 error not >= 8% below V: no VT read (rule)" >> $LOG; fi
echo "S47F_DONE $(date -Is)" >> $LOG
