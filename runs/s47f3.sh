#!/bin/bash
# §47f amended (user, Oct 17 22:47: accelerate for the 03:00 Development close; before any V60 result): seed 3 stopped; V60 = seeds 1-2.
# VALID read V vs V60 on seeds 1-2 (matched seed count); if V60's mean-7 error <= 0.92 x V's, the §45y VT read f17eq vs f17V60 with
# V60 s1-2 at 6.75 each (share 0.25 kept, as with three seeds at 4.5). Same tag (S47Ff17V60), so runs/s47g.sh reads it unchanged.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s47f.log
echo "[s47f] amended to seeds 1-2 (seed 3 stopped for the deadline) $(date -Is)" >> $LOG
until [ -f $L/head_boundary_v8_V60_s1.pt ] && [ -f $L/head_boundary_v8_V60_s2.pt ]; do sleep 30; done; echo "[s47f] V60 s1 s2 done $(date -Is)" >> $LOG
R=$(.venv/bin/python - <<'PY'
import numpy as np
L = np.load("runs/strat/fusion/boundary_labels_v8.npz"); y = L["y_valid"].astype(np.float64)
def err(stem):
    p = np.mean([np.load(f"runs/strat/fusion/head_boundary_v8_{stem}_s{s}.npz")["pred_valid"] for s in (1, 2)], 0); out = []
    for k in range(71):
        m = np.isfinite(y[:, k]); out.append(np.mean((y[m, k] - p[m, k]) ** 2) / np.var(y[m, k]))
    return np.array(out)
a, b = err("V"), err("V60"); r = b[:7].mean() / a[:7].mean()
print(f"{r:.4f} VALID sample 1-R2 mean-7 (seeds 1-2) V {a[:7].mean():.5f} V60 {b[:7].mean():.5f} | rays V {a[7:].mean():.5f} V60 {b[7:].mean():.5f} ratio {b[7:].mean() / a[7:].mean():.3f}")
PY
); echo "[s47f] VALID ratio $R" >> $LOG
if awk -v r="${R%% *}" 'BEGIN{exit !(r <= 0.92)}'; then
  SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2)
  C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
  QPI="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"; QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
  N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"; VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"
  H="route:$N+$QPI+$QHCS+$VV|$R_+$QPI+$QHCS+$VV"
  flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s47f 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S47Ff17V60 --psi psiF \
    --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
    --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"f17V60\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
    || { echo "S47F_FAIL vt $(date -Is)" >> $LOG; exit 1; }
  .venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17eq" "PQmix~g_f17V60" >> $LOG 2>&1
else echo "[s47f] V60 error not >= 8% below V: no VT read (rule)" >> $LOG; fi
echo "S47F_DONE $(date -Is)" >> $LOG
