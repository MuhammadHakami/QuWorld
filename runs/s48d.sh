#!/bin/bash
# §48d (doc 35; registered 2026-10-18 before any §48c search result): what follows S48_DONE.
# 1) winner (§48c rule): the candidate (best | consensus) with the higher mean held-out gain on folds 3-5 that also beats V's held-out gain on
#    >= 2 of 3 folds; none -> stop (§48c closes with V60 as the VisionHOPE head).
# 2) W60 = the winner trained 60 epochs on all TRAIN, seeds 1-2 (two jobs at once), as V60.
# 3) VALID gate: W60's mean-7 1-R2 on the 180-shot VALID sample (seeds 1-2) <= V60's (ratio <= 1.00); else stop.
# 4) VT read (§45y protocol, 150 shots, TRAIN-only heads, paired vs the F18K equivalent PQmix~g_f17V60): arm f19W = F18K with W60 s1-2 added at
#    6.75 each on every shot and the two J1QPIpi heads removed - the only budget-feasible addition (F18K ~23 A100-h; J1QPIpi ~10 A100-h of
#    simulated register for 7.5% of the shape weight; W60 2 seeds ~4-5 A100-h). Adopt as F19K iff the paired 95% CI lower bound > 0.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$PWD/$L/s48d.log   # absolute: the winner step cds into paper/strat/fusion (Oct 18 20:48 bug)
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2"
echo "[s48d] start $(date -Is)" >> $LOG
until grep -q "^S48_DONE" $L/s48.log 2>/dev/null; do sleep 300; done
W=$(cd paper/strat/fusion && ../../../.venv/bin/python - 2>>$LOG <<'PY'
import json
import numpy as np
import hpo_vq as H
r = json.load(open(H.Q / "hpo_vq.json")); hold = r["hold"]; v = [x[0] for x in hold.get("V", [])]; best = None
for nm in ("best", "consensus"):
    g = [x[0] for x in hold.get(nm, [])]
    if len(g) == len(v) == 3 and sum(a > b for a, b in zip(g, v)) >= 2 and (best is None or np.mean(g) > best[1]):
        best = (nm, float(np.mean(g)))
print("NONE" if best is None else best[0] + " " + json.dumps(H.to_p(r["cand"][best[0]])))
PY
)
echo "[s48d] winner: ${W:-ERROR} $(date -Is)" >> $LOG
case "$W" in "") echo "S48D_FAIL winner step produced no output $(date -Is)" >> $LOG; exit 1;; NONE*) echo "S48D_STOP no winner (§48c closes; V60 stays) $(date -Is)" >> $LOG; exit 0;; esac
J=${W#* }
for s in 1 2; do [ -f $L/head_boundary_v8_W60_s$s.pt ] || { env $E QW_GPU_GB=30 runs/capped.sh s48dW$s 40 .venv/bin/python paper/strat/fusion/quota_fusion.py \
  final W60 "$J" $s 60 > $L/s48d_W60_s$s.log 2>&1; echo "[s48d] W60 s$s exit $? $(date -Is)" >> $LOG; } & sleep 60; done; wait
[ -f $L/head_boundary_v8_W60_s1.pt ] && [ -f $L/head_boundary_v8_W60_s2.pt ] || { echo "S48D_FAIL training $(date -Is)" >> $LOG; exit 1; }
R=$(.venv/bin/python - <<'PY'
import numpy as np
L = np.load("runs/strat/fusion/boundary_labels_v8.npz"); y = L["y_valid"].astype(np.float64)
def err(stem):
    p = np.mean([np.load(f"runs/strat/fusion/head_boundary_v8_{stem}_s{s}.npz")["pred_valid"] for s in (1, 2)], 0); out = []
    for k in range(71):
        m = np.isfinite(y[:, k]); out.append(np.mean((y[m, k] - p[m, k]) ** 2) / np.var(y[m, k]))
    return np.array(out)
a, b = err("V60"), err("W60")
print(f"{b[:7].mean() / a[:7].mean():.4f} VALID sample 1-R2 mean-7 V60 {a[:7].mean():.5f} W60 {b[:7].mean():.5f} | rays V60 {a[7:].mean():.5f} W60 {b[7:].mean():.5f}")
PY
); echo "[s48d] VALID ratio $R $(date -Is)" >> $LOG
awk -v r="${R%% *}" 'BEGIN{exit !(r <= 1.0)}' || { echo "S48D_STOP W60 not better than V60 on VALID (rule) $(date -Is)" >> $LOG; exit 0; }
SHOTS=$(grep -o -- '--vt-shots "[0-9,]*"' $L/s44h_jgn_end.sh | cut -d'"' -f2)
C='"shift", "bw", "xhe", "kw", "xp", "jgn", "li", "li2", "kw2"'; X='"xp": {"sigma": 0.15, "cap": 0.1}, "li": {"cap": 0.9}, "xhe": {"gate": 0.0, "w": 0.75}'
QHCS="head_boundary_v8_J1QHCS_s1*4+head_boundary_v8_J1QHCS_s2*4+head_boundary_v8_J1QHCS_s3*4"
N="head_boundary_v8_J1RL_s1*24"; R_="head_boundary_v8_J1RLCS_s1*24"
VW="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75+head_boundary_v8_W60_s1*6.75+head_boundary_v8_W60_s2*6.75"
H="route:$N+$QHCS+$VW|$R_+$QHCS+$VW"
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s48d 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag S48Df19W --psi psiF \
  --psis "PQmix~g" --guard psiF:0.25 --qb qh50 --qbheads "head_qb_QH_s1+head_qb_QH_s2+head_qb_QH_s3+head_qb_QH_s4+head_qb_QH_s5+head_qb_QH_s6" \
  --heads "$H" --xpheads head_xp_XP1CS_s1 --kws "{\"f19W\": {\"_steps\": [$C], $X}}" --axis head --qbclamp --vt-shots "$SHOTS" >> $LOG 2>&1 \
  || { echo "S48D_FAIL vt $(date -Is)" >> $LOG; exit 1; }
.venv/bin/python paper/strat/fusion/vt_pp_compare.py "PQmix~g_f17V60" "PQmix~g_f19W" >> $LOG 2>&1
lo=$(grep -o "PQmix~g_f19W vs PQmix~g_f17V60: S [0-9.]* vs [0-9.]* delta [-+0-9.]* \[[-+0-9.]*" $LOG | tail -1 | sed 's/.*\[//')
awk -v l="$lo" 'BEGIN{exit !(l > 0)}' && echo "S48D_ADOPT F19K = F18K - J1QPIpi + W60 s1-2 (CI lower $lo) $(date -Is)" >> $LOG \
  || echo "S48D_REJECT (CI lower ${lo:-?}; F18K stays) $(date -Is)" >> $LOG
echo "S48D_DONE $(date -Is)" >> $LOG
