#!/bin/bash
# doc 35 §38ad (user Oct 5, fallback): our SoTA head (J1: mlp 4096 x 4, drop 0.1, lr 2.53e-3, 60 epochs) + the 30-atom register
# (stage-2 engagement) + the physics-informed volume constraint (best stage-1 physics weight) -> VALID head screen vs J1.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion
while ! grep -q "S38AC_FINAL_DONE" $L/s38ac_final.log 2>/dev/null; do sleep 60; done
# stage 2 (quantum NAS on stage-1 trial 11; re-run after the 02:34 crash, fixed in hpo_qpikan.py)
flock runs/strat/gpu.lock env QW_GPU_GB=40 runs/capped.sh s38ac 40 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --stage2 11 --study qpikan_q2b \
  --trials 16 --frac 0.25 --epochs 6 >> $L/s38ac_q2.log 2>&1
P=$(.venv/bin/python - <<'PY'
import json, optuna
db = "sqlite:///runs/strat/fusion/hpo_qpikan.db"
v1 = [t for t in optuna.load_study(study_name="qpikan_v1", storage=db).trials if t.value is not None and t.params.get("pi")]
w = max(v1, key=lambda t: t.value).params["w_pi"] if v1 else 0.1
try:
    q2 = optuna.load_study(study_name="qpikan_q2b", storage=db).best_trial.params
except Exception:
    q2 = dict(q_layers=2, q_in=64, link="independent", alpha_mult=10.0, alpha_init=0.1)
p = dict(block="mlp", width=4096, depth=4, drop=0.1, lr=0.0025322139193052723, w_pi=w, q_layers=q2["q_layers"], q_in=q2["q_in"],
         link=q2["link"], alpha_mult=q2["alpha_mult"], alpha_init=q2["alpha_init"])
print(json.dumps(p))
PY
)
echo "S38AD fallback params $P $(date -Is)" >> $L/s38ad_fallback.log
flock runs/strat/gpu.lock env QW_GPU_GB=60 runs/capped.sh s38ad 60 .venv/bin/python paper/strat/fusion/hpo_qpikan.py --fixed "$P" --tag J1QPI \
  --epochs 60 --seed 1 >> $L/s38ad_fallback.log 2>&1
.venv/bin/python paper/strat/fusion/head_screen.py head_boundary_v8_J1_s1 head_boundary_v8_J1QPI_s1 "head_boundary_v8_J1_s1+head_boundary_v8_J1QPI_s1" >> $L/s38ad_fallback.log 2>&1
echo "S38AD_DONE $(date -Is)" >> $L/s38ad_fallback.log
