"""Classical + quantum architecture search for the Sophelio DIII-D row, in two Optuna stages (Sep 23 restructure).

    .venv/bin/python paper/nas/sophelio_nas.py --stage trunk    --trials 20 --study trunk1
    .venv/bin/python paper/nas/sophelio_nas.py --stage coupling --trials 24 --study coupling1 --trunk-study trunk1

State of the art followed (Sep 2026):
  * hybrid quantum-classical NAS searches encoding, ansatz, measurement and the classical-quantum coupling,
    multi-objective with a hardware cost model (Hyb-HANAS, DAC 2026: NSGA-II over accuracy / hardware cost / params;
    HQ-NAS arXiv 2605.18345);
  * Optuna 5.0 (Sep 2026): multivariate TPE default, multi-objective TPE, trial.set_constraint;
  * multi-fidelity successive halving on training epochs (real low-fidelity training replaces zero-cost proxies);
  * LLM-in-the-loop NAS (LLM-NAS 2510.01472): designs from our own research are enqueued as seed trials.
Why two stages: a joint trial that trains a 30-qubit register over all 1.4M frames for 30 epochs took > 4 h on the
shared GB10. Stage 1 searches the classical trunk (the leaderboard lever) with cheap trials; stage 2 searches the
coupling (none / latent register / spectral-routed register, register graph and size) on the FROZEN best stage-1
trunk, training only the coupling on cached trunk latents. That is also the cleaner attribution: the trunk is fixed.
Objectives: stage 1 maximises VALID official S (fixed 40-shot VALID subset, vendored scorer); stage 2 maximises S and
minimises quantum cost atoms x layers under <= 30 simulated atoms. TEST is never touched here; Pareto picks are re-run
with 3 seeds beside matched MLP cores and scored once on TEST.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import optuna

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

OUT = ROOT / "runs" / "nas" / "sophelio"
RUNGS = (4, 12)
FULL_EPOCHS = 30
COUPLING_EPOCHS, COUPLING_STRIDE = 4, 4        # stage 2: coupling-only training on every 4th TRAIN frame

TRUNK_V1 = dict(k=128, width=1024, depth=4, drop=0.05, lr=1e-3, batch=4096, w_scal=1.0, gamma=0.0, lags="sym",
                thomson=True)                      # the full-data v1 trunk (VALID S 0.9822 before the n_e/p_e fix)
TRUNK_SEEDS = [TRUNK_V1, TRUNK_V1 | dict(gamma=0.5), TRUNK_V1 | dict(k=256, width=2048)]
COUPLING_SEEDS = [  # the couplings our research argues for
    dict(quantum="latent", n_reg=3, n_per=10, L=4, geo="grid", sp=1.0, link="shared", lr=1e-3),
    dict(quantum="srr", n_reg=3, n_per=10, L=4, geo="grid", sp=1.0, link="shared", k_lo=32, lr=1e-3),
    dict(quantum="srr", n_reg=2, n_per=10, L=3, geo="ring", sp=0.7, link="shared", k_lo=16, lr=1e-3),
]


def qcfg_of(p: dict) -> str:
    total = p["n_reg"] * p["n_per"]
    return f"q{total}-part={p['n_reg']}x{p['n_per']}-geo={p['geo']}-sp={p['sp']}-link={p['link']}-L={p['L']}"


def space_trunk(trial: optuna.Trial) -> tuple[dict, int]:
    cfg = dict(
        k=trial.suggest_categorical("k", [64, 128, 256]),
        width=trial.suggest_categorical("width", [512, 1024, 2048]),
        depth=trial.suggest_int("depth", 2, 8),
        drop=trial.suggest_categorical("drop", [0.0, 0.05, 0.1]),
        lr=trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        batch=trial.suggest_categorical("batch", [2048, 4096, 8192]),
        w_scal=trial.suggest_float("w_scal", 0.1, 3.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        lags=trial.suggest_categorical("lags", ["none", "back", "sym"]),
        thomson=trial.suggest_categorical("thomson", [True, False]),
        quantum="none", epochs=FULL_EPOCHS)
    trial.set_user_attr("cfg", json.dumps(cfg))
    return cfg, 0


def space_coupling(trial: optuna.Trial, trunk_cfg: dict, init: str) -> tuple[dict, int]:
    cfg = dict(trunk_cfg, init=init, freeze_trunk=True, frame_stride=COUPLING_STRIDE, epochs=COUPLING_EPOCHS,
               lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
               quantum=trial.suggest_categorical("quantum", ["latent", "srr"]))
    p = dict(n_reg=trial.suggest_int("n_reg", 1, 3), n_per=trial.suggest_categorical("n_per", [5, 8, 10]),
             L=trial.suggest_categorical("L", [2, 3, 4, 6]), geo=trial.suggest_categorical("geo", ["grid", "chain", "ring", "tri"]),
             sp=trial.suggest_categorical("sp", [0.7, 1.0, 1.5]), link=trial.suggest_categorical("link", ["independent", "shared"]))
    if p["n_reg"] == 1:
        p["link"] = "independent"
    cfg["qcfg"], cfg["q_layers"] = qcfg_of(p), p["L"]
    if cfg["quantum"] == "srr":
        cfg["k_lo"] = trial.suggest_categorical("k_lo", [8, 16, 32, 64])
    atoms = p["n_reg"] * p["n_per"]
    trial.set_user_attr("atoms", atoms); trial.set_user_attr("cfg", json.dumps(cfg))
    trial.set_constraint("atoms_le_30", float(atoms - 30))        # feasible iff <= 30 simulated atoms (Optuna 5 API)
    return cfg, atoms * p["L"]


def open_study(name: str, directions: list[str], seeds: list[dict]) -> optuna.Study:
    study = optuna.create_study(study_name=f"sophelio-{name}", storage=f"sqlite:///{OUT}/{name}.db", directions=directions,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]   # killed mid-trial (crash/reboot)
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        study.enqueue_trial(dict(t.params), skip_if_exists=False)   # a re-queued stale trial matches its FAILED copy
    if not study.trials:
        for s in seeds:
            study.enqueue_trial(s, skip_if_exists=True)
    return study


def best_trunk(trunk_study: str) -> tuple[dict, str, float]:
    s = optuna.load_study(study_name=f"sophelio-{trunk_study}", storage=f"sqlite:///{OUT}/{trunk_study}.db")
    t = s.best_trial
    return json.loads(t.user_attrs["cfg"]), str(OUT / "trunks" / f"{trunk_study}_t{t.number}.pt"), t.value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["trunk", "coupling"], required=True)
    ap.add_argument("--trials", type=int, default=20); ap.add_argument("--study", required=True)
    ap.add_argument("--trunk-study", default="trunk1"); ap.add_argument("--score-val", type=int, default=40)
    a = ap.parse_args()
    (OUT / "trunks").mkdir(parents=True, exist_ok=True)
    sota.build_cache()
    data = sota.Data()
    if a.stage == "trunk":
        study = open_study(a.study, ["maximize"], TRUNK_SEEDS)
        trunk_cfg = init = None
    else:
        trunk_cfg, init, s_trunk = best_trunk(a.trunk_study)
        print(f"[nas] coupling stage on trunk {init} (VALID S {s_trunk:.4f})", flush=True)
        study = open_study(a.study, ["maximize", "minimize"], COUPLING_SEEDS)
    best_proxy_at = {}                                    # rung -> proxies seen (successive halving, keep top half)
    for t in study.trials:                                # resume: rebuild the halving history from stored proxies
        for key, val in t.user_attrs.items():
            if key.startswith("proxy_ep"):
                best_proxy_at.setdefault(int(key[len("proxy_ep"):]), []).append(val)

    def objective(trial: optuna.Trial):
        cfg, qcost = space_trunk(trial) if a.stage == "trunk" else space_coupling(trial, trunk_cfg, init)

        def report(epoch, proxy):
            seen = best_proxy_at.setdefault(epoch, [])
            seen.append(proxy)
            trial.set_user_attr(f"proxy_ep{epoch}", proxy)
            if len(seen) >= 4 and proxy < sorted(seen, reverse=True)[len(seen) // 2]:
                raise optuna.TrialPruned(f"epoch {epoch} proxy {proxy:.4f} below the running median")

        save = OUT / "trunks" / f"{a.study}_t{trial.number}.pt" if a.stage == "trunk" else None
        res = sota.train_eval(cfg, data, report=report if a.stage == "trunk" else None,
                              rungs=RUNGS if a.stage == "trunk" else (), score_val=a.score_val, save=save)
        v = res["valid_official"]
        for key in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency"):
            trial.set_user_attr(f"val_{key}", v.get(key))
        trial.set_user_attr("r2_cons_each", json.dumps(v.get("r2_cons_each")))
        trial.set_user_attr("params", res["params"]); trial.set_user_attr("trainable", res["trainable_params"])
        trial.set_user_attr("train_seconds", res["train_seconds"])
        print(f"[nas] {a.stage} trial {trial.number} {cfg['quantum']} {cfg.get('qcfg')} -> VALID S {v['S']:.4f} "
              f"(psi {v['r2_psi']:.4f} qb {v['r2_qb']:.4f} cons {v['consistency']:.4f}) qcost {qcost} "
              f"{res['train_seconds']/60:.0f} min", flush=True)
        return v["S"] if a.stage == "trunk" else (v["S"], qcost)

    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, a.trials - done), catch=(RuntimeError,))
    if a.stage == "trunk":
        b = study.best_trial
        (OUT / f"{a.study}_best.json").write_text(json.dumps(dict(number=b.number, S=b.value, cfg=json.loads(b.user_attrs["cfg"])), indent=1))
        print(f"[nas] trunk best: trial {b.number} VALID S {b.value:.4f}", flush=True)
    else:
        front = [dict(number=t.number, S=t.values[0], qcost=t.values[1], cfg=json.loads(t.user_attrs["cfg"])) for t in study.best_trials]
        (OUT / f"{a.study}_front.json").write_text(json.dumps(front, indent=1))
        print(f"[nas] Pareto front: {[(f['number'], round(f['S'], 4), f['qcost']) for f in front]}", flush=True)


if __name__ == "__main__":
    main()
