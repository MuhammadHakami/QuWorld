"""Optuna HPO driver (user directive Sep 21 2026): tune every arm on TRAIN/VALID with the same trial
budget, then evaluate the selected configuration ONCE on TEST with 3 seeds.

    uv run --no-sync python paper/hpo/driver.py <bench> <arm> search --trials N [--seed 1337]
    uv run --no-sync python paper/hpo/driver.py <bench> <arm> final  [--seeds 1337 1338 1339]

A benchmark module `paper/hpo/bench_<bench>.py` provides:
    ARMS: dict arm -> base variant string (e.g. {"none": "none", "mlp": "q30-mlpref", "q": "q30"})
    space(trial, arm) -> dict of hyperparameters (shared optimiser/schedule keys for every arm; register
                         keys only for the register arm; nothing budget-changing for the control)
    run(variant, hp, seed, split) -> dict with "metric" (lower is better) on `split` in {"valid", "test"},
                                     plus whatever else should be logged (params, seconds, extra metrics)
Studies persist in runs/hpo/<bench>/<arm>.db (resumable; TPE with a fixed sampler seed). The search never
reads the test split; `final` is the only stage that does, and it writes runs/hpo/<bench>/<arm>_final.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import time

import optuna

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT / "paper" / "hpo"))


def study_for(bench: str, arm: str, seed: int) -> optuna.Study:
    out = ROOT / "runs" / "hpo" / bench
    out.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=f"{bench}-{arm}", storage=f"sqlite:///{out}/{arm}.db",
                               direction="minimize", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5))


def variant_for(mod, arm: str, hp: dict) -> str:
    """Compose the variant string from the arm's base variant and register hyperparameters."""
    base = mod.ARMS[arm]
    if not base.startswith("q"):
        return base
    head, *opts = base.split("-")
    reg = {k: v for k, v in hp.items() if k in ("geo", "sp", "link", "L", "part")}
    parts = [head] + [f"{k}={v}" for k, v in reg.items()] + opts
    return "-".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("arm"); ap.add_argument("stage", choices=["search", "final"])
    ap.add_argument("--trials", type=int, default=16); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--search-seeds", type=int, nargs="+", default=None,
                    help="objective = mean VALID metric over these seeds (multi-seed selection; default: the single --seed)")
    ap.add_argument("--study", default=None, help="study name suffix (e.g. ms2) to keep multi-seed studies separate")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 1338, 1339])
    a = ap.parse_args()
    mod = importlib.import_module(f"bench_{a.bench}")
    out = ROOT / "runs" / "hpo" / a.bench
    arm_key = a.arm + (f"_{a.study}" if a.study else "")
    study = study_for(a.bench, arm_key, a.seed)
    search_seeds = a.search_seeds or [a.seed]
    if a.stage == "search":
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"[hpo] {a.bench}/{a.arm}: {done} complete trials on file, target {a.trials}", flush=True)

        def objective(trial: optuna.Trial) -> float:
            hp = mod.space(trial, a.arm)
            variant = variant_for(mod, a.arm, hp)
            t0 = time.time()
            per_seed = [mod.run(variant, hp, sd, "valid") for sd in search_seeds]
            res = dict(per_seed[0]); res["metric"] = float(sum(r["metric"] for r in per_seed) / len(per_seed))
            res["metric_per_seed"] = [r["metric"] for r in per_seed]
            trial.set_user_attr("variant", variant); trial.set_user_attr("search_seeds", search_seeds)
            for k, v in res.items():
                if k not in ("metric", "metric_per_seed") and isinstance(v, (int, float, str)):
                    trial.set_user_attr(k, v)
            print(f"[hpo] trial {trial.number} {variant} {hp} -> valid {res['metric']:.6f} ({time.time()-t0:.0f}s)", flush=True)
            return float(res["metric"])

        while done < a.trials:
            study.optimize(objective, n_trials=1)
            done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        best = study.best_trial
        summary = dict(bench=a.bench, arm=a.arm, trials=done, best_value=best.value, best_params=best.params,
                       best_variant=best.user_attrs.get("variant"), search_seed=a.seed)
        (out / f"{arm_key}_search.json").write_text(json.dumps(summary, indent=1))
        print(f"[hpo] {a.bench}/{a.arm} search done: best valid {best.value:.6f} {best.params}", flush=True)
    else:
        best = study.best_trial
        hp, variant = best.params, best.user_attrs.get("variant")
        results = []
        for seed in a.seeds:
            res = mod.run(variant, hp, seed, "test")
            res.update(seed=seed, variant=variant, hp=hp)
            results.append(res)
            print(f"[hpo] final {a.bench}/{a.arm} seed {seed}: test {res['metric']:.6f}", flush=True)
        vals = [r["metric"] for r in results]
        import statistics as st
        summary = dict(bench=a.bench, arm=a.arm, variant=variant, hp=hp, best_valid=best.value,
                       test_mean=st.mean(vals), test_sd=(st.stdev(vals) if len(vals) > 1 else 0.0), per_seed=results)
        (out / f"{arm_key}_final.json").write_text(json.dumps(summary, indent=1))
        print(f"[hpo] {a.bench}/{a.arm} FINAL test {summary['test_mean']:.6f} ± {summary['test_sd']:.6f}", flush=True)


if __name__ == "__main__":
    main()
