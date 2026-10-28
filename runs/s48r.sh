#!/bin/bash
# §48c resumed after the F18K Development build (doc 35, 2026-10-18; the runner and its 3 workers were stopped at 00:38 so the public build
# ran alone). Trials stopped in flight (the enqueued references, state RUNNING) are marked FAIL and re-enqueued with the same parameters and
# register; then as runs/s48.sh from step 2: search to 20 COMPLETE + PRUNED trials on folds 0-2, confirm V / best / consensus on folds 3-5, report.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PATH=$PWD/.venv/bin:$PATH; L=runs/strat/fusion; LOG=$L/s48.log
E="QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 QW_GPU_GB=24"; P=paper/strat/fusion/hpo_vq.py
echo "[s48] resume $(date -Is)" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.load_study(study_name="s48_vq", storage="sqlite:///runs/strat/fusion/quota/hpo_vq.db")
stale = [t for t in st.trials if t.state == TrialState.RUNNING]
for t in stale:
    st._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
    st.enqueue_trial(t.system_attrs.get("fixed_params") or t.params, user_attrs={"reg": t.user_attrs.get("reg", "tether3d")})
print(f"[s48] resume: {len(stale)} stopped trials -> FAIL and re-enqueued: {[t.number for t in stale]}")
PY
for w in 0 1 2; do env $E HPO_WORKER=$w runs/capped.sh s48w$w 36 .venv/bin/python $P search 20 > $L/s48_w${w}r.log 2>&1 & sleep 90; done; wait
echo "[s48] search done $(date -Is)" >> $LOG; .venv/bin/python $P report >> $LOG 2>&1
for c in V best consensus; do env $E runs/capped.sh s48k$c 36 .venv/bin/python $P confirm $c 3,4,5 > $L/s48_conf_$c.log 2>&1 & sleep 20; done; wait
.venv/bin/python $P report >> $LOG 2>&1
echo "S48_DONE $(date -Is)" >> $LOG
