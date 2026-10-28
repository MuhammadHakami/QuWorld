"""TDC ADMET ensembles: MapLight (GPU CatBoost) + CheMeleon arm, per official seed, scored by TDC's evaluator.

    .venv-tdc/bin/python paper/tdc/ensemble.py   ->  runs/tdc_ensemble/summary.json (+ printed table)

Pre-declared (no weight tuning; MapLight trains on the whole train_val, so there is no held-out split to tune on):
seed s of the ensemble = the equal-weight mean of MapLight seed s and CheMeleon-<arm> seed s (probabilities for binary
endpoints, values for regression). Arms: vanilla (classical ensemble), mlp30 (matched classical core), mr30ryd
(register). The register-vs-MLP comparison therefore stays matched inside the same ensemble.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tdcdata as D  # noqa: E402

ML, CM, OUT = D.ROOT / "runs" / "tdc_maplight", D.ROOT / "runs" / "tdc_chemeleon", D.ROOT / "runs" / "tdc_ensemble"
ML2 = D.ROOT / "runs" / "tdc_maplight_chemeleon"
ARMS = ("vanilla", "mlp30", "mr30ryd")


def load(path: pathlib.Path) -> dict | None:
    return dict(np.load(path)) if path.exists() else None


def score(g, name: str, metric: str, preds: dict) -> list[float]:
    return [float(g.evaluate({name: preds[k]})[name][metric]) for k in sorted(preds)]


def main() -> None:
    g = D.group(); OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for ep in D.endpoints():
        name, metric = g.get(ep)["name"], D.METRICS[ep]
        ml = load(ML / f"{ep}_preds.npz")
        row = {"metric": metric, "higher_is_better": D.higher_is_better(ep)}
        if ml is not None:
            row["maplight"] = score(g, name, metric, ml)
        ml2 = load(ML2 / f"{ep}_preds.npz")
        if ml2 is not None:
            row["maplight_chem"] = score(g, name, metric, ml2)
        for arm in ARMS:
            cm = load(CM / f"{ep}_{arm}_preds.npz")
            if cm is None:
                continue
            row[f"chemeleon_{arm}"] = score(g, name, metric, cm)
            if ml is not None and set(ml) == set(cm):
                row[f"ens_{arm}"] = score(g, name, metric, {k: 0.5 * (ml[k] + cm[k]) for k in ml})
            if ml2 is not None and set(ml2) == set(cm):
                row[f"ens2_{arm}"] = score(g, name, metric, {k: 0.5 * (ml2[k] + cm[k]) for k in ml2})
        rows[ep] = row
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    cols = ["maplight", "maplight_chem"] + [f"chemeleon_{a}" for a in ARMS] + [f"ens_{a}" for a in ARMS] + [f"ens2_{a}" for a in ARMS]
    print(f"{'endpoint':32s} {'metric':9s} " + " ".join(f"{c:>15s}" for c in cols))
    for ep, r in rows.items():
        cell = lambda c: f"{np.mean(r[c]):.3f}±{np.std(r[c]):.3f}" if c in r else "—"
        print(f"{ep:32s} {r['metric']:9s} " + " ".join(f"{cell(c):>15s}" for c in cols))


if __name__ == "__main__":
    main()
