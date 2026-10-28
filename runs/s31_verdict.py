"""Doc 35 §31 / §31e verdict on the full TEST split (each frozen model scored once by coproc.py --score_test).

Registered rule (§31 step 3, prediction locked in runs/s31/predictability.json): with S >= 0.01 the selected routing
(proxy on both rows) is predicted to beat pooled insertion on full TEST in BOTH the register and the matched-MLP arms
(positive paired relative VRMSE gain); a predicted-positive row with either core's mean TEST gain <= 0 is a failure.
Reported per row and core: per-seed paired gains 1 - VRMSE(proxy)/VRMSE(pooled), their mean, a 2,000-draw trajectory-
cluster bootstrap 95% interval of the seed-averaged gain, the register-minus-MLP difference (descriptive), the
official The Well convention (variance with Bessel's correction: VRMSE x sqrt((n-1)/n), n pixels per field), and runtime
and parameters from the phase-A runs.

    .venv/bin/python runs/s31_verdict.py   ->  runs/s31/test_verdict.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "runs/alloc/results"
PRED = json.loads((ROOT / "runs/s31/predictability.json").read_text())
SEEDS, CORES = (1, 2, 3), ("ryd", "mlp")


def load(row: str, split: str, st: str, core: str, seed: int) -> dict:
    return json.loads((R / row / split / f"{st}_{core}_K30_s{seed}_s31e.json").read_text())      # §31e arm tag


def cluster_ci(row: str, core: str, route: str, n: int = 2000) -> list[float]:
    """Trajectory-cluster bootstrap of the SAME statistic as mean_gain: mean over seeds of 1 - mean_w(route_s) /
    mean_w(pooled_s), recomputed on each resampled set of TEST trajectories (Codex audit 035507)."""
    res = {(st, s): load(row, "test", st, core, s) for st in (route, "pooled") for s in SEEDS}
    traj = np.asarray(res[(route, 1)]["test_trajectory"])
    assert all(r["test_trajectory"] == res[(route, 1)]["test_trajectory"] for r in res.values()), "window order differs"
    pro = np.asarray([res[(route, s)]["test_per_window"] for s in SEEDS])          # (3, N)
    poo = np.asarray([res[("pooled", s)]["test_per_window"] for s in SEEDS])
    ids = np.unique(traj); members = {t: np.flatnonzero(traj == t) for t in ids}; g = np.random.default_rng(0); draws = []
    for _ in range(n):
        rows = np.concatenate([members[t] for t in g.choice(ids, len(ids), replace=True)])
        draws.append(np.mean(1 - pro[:, rows].mean(1) / poo[:, rows].mean(1)))
    return [round(float(q), 5) for q in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    out = {}
    for row, p in PRED.items():
        route, npx = p["route_lin"], None
        rec = {"S": p["S"], "prediction": p["prediction"], "route": route, "cores": {}}
        for core in CORES:
            gains = [1 - load(row, "test", route, core, s)["test"] / load(row, "test", "pooled", core, s)["test"] for s in SEEDS]
            a = load(row, "test", route, core, 1)
            rec["cores"][core] = {
                "seed_gain": [round(x, 5) for x in gains], "mean_gain": round(float(np.mean(gains)), 5),
                "ci95_cluster": cluster_ci(row, core, route),
                "test_route": [round(load(row, "test", route, core, s)["test"], 6) for s in SEEDS],
                "test_pooled": [round(load(row, "test", "pooled", core, s)["test"], 6) for s in SEEDS],
                "trunk_test": round(a["trunk_test"], 6), "n_test": a["n_test"],
                "params_core": load(row, "vsplit", route, core, 1)["params_core"],
                "seconds_phaseA": [round(load(row, "vsplit", st, core, s)["seconds"], 1) for st in ("pooled", route) for s in SEEDS]}
        shape = load(row, "test", route, "ryd", 1).get("shape")                  # [H, W] of the TEST cache
        if shape:
            npx = int(np.prod(shape)); f = np.sqrt((npx - 1) / npx)
            for c in rec["cores"].values():
                c["test_route_official"] = [round(v * f, 6) for v in c["test_route"]]
        rec["reg_minus_mlp_mean_gain"] = round(rec["cores"]["ryd"]["mean_gain"] - rec["cores"]["mlp"]["mean_gain"], 5)
        pos = p["S"] >= 0.01
        rec["verdict"] = ("prediction held" if all(c["mean_gain"] > 0 for c in rec["cores"].values()) else "prediction failed") \
            if pos else ("abstain" if p["S"] > 0 else ("held" if all(c["mean_gain"] < 0.01 for c in rec["cores"].values()) else "failed"))
        out[row] = rec
    (ROOT / "runs/s31/test_verdict.json").write_text(json.dumps(out, indent=1))
    print("[s31] verdict", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
