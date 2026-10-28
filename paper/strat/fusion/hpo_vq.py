"""Doc 35 §48c: 8-fold stratified-CV search over the VisionHOPE + input-dependent 30-atom register hybrid (quota_fusion V arms).

User Oct 17: "6-8 CVs in one", search on 4 of 8 folds, then aggregate. Data = the §47 stratified half of TRAIN, 8 shot folds
(head_boundary.cv_shots: Ip sign x shot-mean-li quartile strata). Every fold-run trains on 7 folds (20 epochs, seed 1) and predicts
the held-out one. Objective = the Lean CompetitionBlend gain (formal/QuotaScaling/CompetitionBlend.lean, blend_best_gain): mean over
the 7 consistency scalars of R2(best blend with the control M) - R2(M) - it rewards a lower own error B AND a lower error alignment
C, which is what the deployed ensemble gains from (ScoreCeiling.lean / §48b: consistency holds ~63 % of the reducible gap).

  ctrl <folds>      M (the §47 control config) on the given folds -> quota/M8_f<k>.npz
  search <n_total>  one Optuna worker (TPE, shared SQLite study; MedianPruner on the running fold mean): each trial runs folds 0-3
                    in order, pruned after any fold where its running mean is below the median of earlier trials at that fold;
                    V (the §47 config), V4, V4t (§47g) and Vd (x, y, z + tethers) are enqueued first as references; searched trials use
                    the x, y, z tethered register with physical noise (q_noise), input noise (in_noise) and Huber (huber) as dimensions
  confirm <cand> <folds>  one candidate (V | best | consensus) on held-out folds 4-7 -> quota/H8_<cand>_f<k>.npz
  report            aggregate: best = highest mean on folds 0-3 (completed trials); consensus = top-3 trials' log-mean of the
                    continuous parameters and mode of the categorical ones; decision on folds 4-7 (never used to select)

    env <heads env> QW_GPU_GB=30 runs/capped.sh hpo 40 .venv/bin/python paper/strat/fusion/hpo_vq.py search 16
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

import fx  # noqa: I001
import quota_blend as QB
import quota_fusion as QF

Q = fx.OUT / os.environ.get("HPO_DIR", "quota"); Q.mkdir(exist_ok=True); K = 8; SEARCH, HOLD = (0, 1, 2), (3, 4, 5)        # amended Oct 17 22:50 (user: accelerate): 3 search + 3 held-out folds of 8
EPOCHS = int(os.environ.get("HPO_EPOCHS", "20"))             # HPO_DIR / HPO_EPOCHS: smoke runs only
DB = f"sqlite:///{Q / 'hpo_vq.db'}"; STUDY = "s48_vq"
V = {"backbone": "vhope", "seq": True, "crop": 32, "shots": 128, "bb_width": 128, "bb_depth": 2, "tokens": 8}
CROPS = {"16x256": (16, 256), "32x128": (32, 128), "64x64": (64, 64), "128x32": (128, 32)}   # 4096 frames per step in every option
_R = {"bb_depth": 2, "bb_width": 128, "mixer_half": False, "tokens": 8, "crop_shots": "32x128", "lr": QF.BASE["lr"], "drop": 0.1,
      "q_layers": 4, "q_in": 64, "alpha_init": 0.1, "q_noise": 0.0, "in_noise": 0.0, "huber": 0.0}
REFS = [({**_R}, "multi"),                                                   # V   (the §47 config)
        ({**_R, "bb_depth": 4}, "multi"),                                    # V4  (§47g)
        ({**_R, "bb_depth": 4, "mixer_half": True}, "multi"),                # V4t (§47g: VisionHOPE-T stage-1 dims)
        ({**_R}, "tether3d")]                                                # Vd  (§47e: x, y, z + tethers, no noise)
CAT = ["bb_depth", "bb_width", "mixer_half", "tokens", "crop_shots", "drop", "q_layers", "q_in", "q_noise", "in_noise", "huber"]
LOG = ["lr", "alpha_init"]


def setup():
    import head_boundary as HB
    import hpo_qpikan as H
    if "QW_GPU_GB" in os.environ:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["QW_GPU_GB"]) / 120.0))
    d, sota, c, fm, edge, lab = H.data(); tr, Y = lab["rows_train"], lab["y_train"].astype(np.float64)
    half = HB.cv_shots(d, tr, Y, 0, 2); m = np.isin(d.shot[tr], half); tr, Y = tr[m], Y[m]
    folds = [HB.cv_shots(d, tr, Y, k, K) for k in range(K)]
    return d, sota, c, fm, edge, tr, Y, folds


def to_p(h: dict) -> dict:
    """Optuna parameters -> quota_fusion config."""
    cr, sh = {**CROPS, "8x512": (8, 512)}[h["crop_shots"]]                     # §48g: 8x512 = boundary extension (not in the §48c box)
    p = {**QF.BASE, **V, "bb_depth": h["bb_depth"], "bb_width": h["bb_width"], "tokens": h["tokens"], "crop": cr, "shots": sh,
         "lr": h["lr"], "drop": h["drop"], "q_layers": h["q_layers"], "q_in": h["q_in"], "alpha_init": h["alpha_init"],
         "q_noise": h.get("q_noise", 0.0), "in_noise": h.get("in_noise", 0.0), "huber": h.get("huber", 0.0)}
    if h["mixer_half"]:
        p["mixer"] = h["bb_width"] // 2
    if h.get("reg", "tether3d") != "multi":                                   # the search register = x, y, z + tethers (user, Oct 17)
        p["reg"] = h.get("reg", "tether3d")
    return p


def fold_run(D, p, k, tag, seq=True):
    d, sota, c, fm, edge, tr, Y, folds = D; hm = np.isin(d.shot[tr], folds[k]); t0 = time.time()
    fit = QF.train_seq if seq else QF.train_eval
    net, S, P = fit({**p, "_fold": None}, d, sota, c, fm, edge, tr[~hm], Y[~hm], tr[hm], EPOCHS, seed=1, log=f" {tag} f{k}")
    np.savez(Q / f"{tag}_f{k}.npz", rows=tr[hm], pred=P.astype(np.float32), y=Y[hm].astype(np.float32))
    single = float(np.mean(QF.r2_each(Y[hm], P)))
    rb = (2 * net.alpha.detach().abs() * net.q_out.weight.detach().abs().sum(1)).float().cpu().numpy()[:7]   # NoiseRobust.register_shift_le
    del net, S; gc.collect(); torch.cuda.empty_cache()
    print(f"[hpo] {tag} fold {k} single mean7 {single:.5f} | certified register-noise bound (std units, mean-7) {rb.mean():.4f} "
          f"{time.time() - t0:.0f}s", flush=True)
    return single


def gain(tag, k):
    zc, za = np.load(Q / f"M8_f{k}.npz"), np.load(Q / f"{tag}_f{k}.npz"); assert (zc["rows"] == za["rows"]).all()
    s = QB.stats(zc["y"], zc["pred"], za["pred"])
    return float(np.mean([t["r2_best"] - t["r2_ctrl"] for t in s])), float(np.mean([t["r2_arm"] for t in s])), float(np.mean([t["rho"] for t in s]))


def search(D, n_total):
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    st = optuna.create_study(study_name=STUDY, storage=DB, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=int(os.environ.get("HPO_WORKER", "0")), multivariate=True),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0))
    if os.environ.get("HPO_WORKER", "0") == "0" and not st.trials:
        for r, reg in REFS:
            st.enqueue_trial(r, user_attrs={"reg": reg})

    def objective(trial):
        h = {"bb_depth": trial.suggest_categorical("bb_depth", [2, 3, 4]), "bb_width": trial.suggest_categorical("bb_width", [96, 128, 160]),
             "mixer_half": trial.suggest_categorical("mixer_half", [False, True]), "tokens": trial.suggest_categorical("tokens", [4, 8, 12]),
             "crop_shots": trial.suggest_categorical("crop_shots", list(CROPS)), "lr": trial.suggest_float("lr", 4e-4, 3e-3, log=True),
             "drop": trial.suggest_categorical("drop", [0.0, 0.1, 0.2]), "q_layers": trial.suggest_categorical("q_layers", [2, 4, 6]),
             "q_in": trial.suggest_categorical("q_in", [32, 64, 96]), "alpha_init": trial.suggest_float("alpha_init", 0.03, 0.3, log=True),
             "q_noise": trial.suggest_categorical("q_noise", [0.0, 0.02, 0.05]), "in_noise": trial.suggest_categorical("in_noise", [0.0, 0.03, 0.1]),
             "huber": trial.suggest_categorical("huber", [0.0, 0.3, 1.0]), "reg": trial.user_attrs.get("reg", "tether3d")}
        trial.set_user_attr("reg", h["reg"])
        p = to_p(h); tag = f"H8t{trial.number}"; gs = []
        for i, k in enumerate(SEARCH):
            try:
                single = fold_run(D, p, k, tag)
            except torch.OutOfMemoryError:                                    # too large for its memory cap: FAIL this trial only
                gc.collect(); torch.cuda.empty_cache(); raise
            g, _, rho = gain(tag, k); gs.append(g)
            trial.set_user_attr(f"f{k}", {"gain": g, "single": single, "rho": rho})
            print(f"HPO_FOLD t{trial.number} f{k} gain {g:+.5f} single {single:.5f} rho {rho:.3f} | running {np.mean(gs):+.5f} | {json.dumps(h)}", flush=True)
            trial.report(float(np.mean(gs)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(gs))
    st.optimize(objective, callbacks=[MaxTrialsCallback(n_total, states=(TrialState.COMPLETE, TrialState.PRUNED))], gc_after_trial=True,
                catch=(torch.OutOfMemoryError,))


def candidates():
    import optuna
    st = optuna.load_study(study_name=STUDY, storage=DB)
    done = sorted([t for t in st.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
    pr = lambda t: {"q_noise": 0.0, "in_noise": 0.0, "huber": 0.0, **t.params, "reg": t.user_attrs.get("reg", "tether3d")}   # noqa: E731
    best = pr(done[0]); top = done[:3]
    cons = {k: Counter(pr(t)[k] for t in top).most_common(1)[0][0] for k in CAT}
    cons.update({k: float(math.exp(np.mean([math.log(t.params[k]) for t in top]))) for k in LOG})
    cons["reg"] = Counter(pr(t)["reg"] for t in top).most_common(1)[0][0]
    return {"V": {**REFS[0][0], "reg": "multi"}, "best": best, "consensus": cons}, done


def main() -> None:
    mode = sys.argv[1]
    if mode == "report":
        cand, done = candidates()
        for t in done[:6]:
            print(f"HPO_TOP t{t.number} mean {t.value:+.5f} " + " ".join(f"f{k} {t.user_attrs[f'f{k}']['gain']:+.5f}" for k in SEARCH) + f" | {t.params}")
        res = {}
        for nm in cand:
            rows = [gain(f"H8_{nm}", k) for k in HOLD if (Q / f"H8_{nm}_f{k}.npz").exists()]
            res[nm] = rows
            if rows:
                print(f"HPO_HOLD {nm}: mean gain {np.mean([r[0] for r in rows]):+.5f} single {np.mean([r[1] for r in rows]):.5f} rho "
                      f"{np.mean([r[2] for r in rows]):.3f} over {len(rows)} held-out folds | {cand[nm]}", flush=True)
        (Q / "hpo_vq.json").write_text(json.dumps({"cand": cand, "hold": res}, indent=1, default=float)); return
    D = setup()
    if mode == "ctrl":
        for k in map(int, sys.argv[2].split(",")):
            if not (Q / f"M8_f{k}.npz").exists():
                fold_run(D, {**QF.BASE, "_fold": None}, k, "M8", seq=False)
    elif mode == "search":
        search(D, int(sys.argv[2]))
    elif mode == "probe":                                                     # §48g: one-factor boundary extensions around the search winner
        nm, over = sys.argv[2], json.loads(sys.argv[3]); cand, _ = candidates(); p = to_p({**cand["best"], **over})
        for k in map(int, sys.argv[4].split(",")):
            if not (Q / f"H8_{nm}_f{k}.npz").exists():
                fold_run(D, p, k, f"H8_{nm}"); print(f"HPO_PROBE {nm} f{k} gain {gain(f'H8_{nm}', k)[0]:+.5f}", flush=True)
    elif mode == "confirm":
        nm = sys.argv[2]; cand, _ = candidates(); p = to_p(cand[nm])
        for k in map(int, sys.argv[3].split(",")):
            if not (Q / f"H8_{nm}_f{k}.npz").exists():
                fold_run(D, p, k, f"H8_{nm}"); print(f"HPO_CONFIRM {nm} f{k} gain {gain(f'H8_{nm}', k)[0]:+.5f}", flush=True)


if __name__ == "__main__":
    main()
