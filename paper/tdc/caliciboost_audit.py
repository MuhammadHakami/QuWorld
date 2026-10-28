"""TDC Caco-2: CaliciBoost (TDC leader, MAE 0.256) reproduced, and audited for test-informed feature selection.

CaliciBoost (env .venv-calici = its pins: sklearn 1.2.2, xgboost 1.7.3, numpy 1.26.4, pandas 2.2.3; github.com/Calici/CaliciBoost @ a11251f, vendored in baseline/CaliciBoost): released PaDEL descriptors,
StandardScaler on train_val, a hard-coded per-seed feature subset, XGBoost (its params, max_bin 512). TDC's train_val and
test are identical across seeds, yet the subsets differ by seed and no validation split exists in the code, so the subsets
may have been chosen with test feedback. Arms (5 seeds, official evaluate_many):
  exact     the released subsets (CPU hist instead of gpu_hist; the only change)
  *_gpu     the same arm with the released gpu_hist call (3080 Ti); exact_gpu is the released call itself
  inner     leakage-free: the same count of features per seed, chosen by permutation importance (MAE) on 5-fold CV inside
            train_val only (KFold shuffled by the seed), then the same XGBoost on all of train_val
  all       no selection: every PaDEL descriptor
Per-seed test predictions are saved for ensembles.

    .venv-calici/bin/python paper/tdc/caliciboost_audit.py [--arms exact inner all] [--jobs 16]   -> runs/tdc_calici/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = pathlib.Path(__file__).resolve().parents[2]
CB = ROOT / "baseline" / "CaliciBoost"
sys.path.insert(0, str(CB))
from calici_boost import clean_data, feature_candidates, xg_parmas  # noqa: E402

OUT = ROOT / "runs" / "tdc_calici"
SEEDS = (1, 2, 3, 4, 5)


def data():
    from tdc.benchmark_group import admet_group
    group = admet_group(path=str(ROOT / "data" / "tdc"))
    b = group.get("Caco2_Wang"); desc = pd.read_csv(CB / "PaDEL_Descriptors.csv")
    tr = pd.merge(b["train_val"], desc, on=["Drug_ID", "Drug"], how="left")
    te = pd.merge(b["test"], desc, on=["Drug_ID", "Drug"], how="left")
    tr, te = clean_data(tr, te)
    return group, b["name"], tr, te


def model(seed, jobs, gpu=False):
    if gpu:                                                                   # exactly the released call (gpu_hist)
        return XGBRegressor(**xg_parmas, max_bin=512, random_state=seed, tree_method="gpu_hist", gpu_id=0)
    return XGBRegressor(**xg_parmas, max_bin=512, random_state=seed, tree_method="hist", n_jobs=jobs)


def inner_select(x, y, k, seed, jobs, gpu=False):
    """Top-k features by permutation importance (MAE) averaged over 5 inner folds of train_val."""
    imp = np.zeros(x.shape[1])
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(x):
        m = model(seed, jobs, gpu).fit(x[tr], y[tr])
        r = permutation_importance(m, x[va], y[va], scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed, n_jobs=1)
        imp += r.importances_mean
    return np.argsort(-imp)[:k]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", default=["exact", "inner", "all"])
    ap.add_argument("--jobs", type=int, default=16); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); group, name, tr, te = data()
    x_tr, y_tr = tr.iloc[:, 3:].to_numpy(float), tr.iloc[:, 2].to_numpy(float)
    x_te = te.iloc[:, 3:].to_numpy(float); print(f"[calici] train {x_tr.shape} test {x_te.shape}", flush=True)
    sc = StandardScaler().fit(x_tr); x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
    rep = {}
    for arm in a.arms:
        preds, sel, t = [], {}, time.time()
        for seed in SEEDS:
            base, gpu = arm.removesuffix("_gpu"), arm.endswith("_gpu")
            if base == "exact":
                idx = np.asarray(feature_candidates[seed])
            elif base == "inner":
                idx = inner_select(x_tr, y_tr, len(feature_candidates[seed]), seed, a.jobs, gpu)
            else:
                idx = np.arange(x_tr.shape[1])
            p = model(seed, a.jobs, gpu).fit(x_tr[:, idx], y_tr).predict(x_te[:, idx])
            preds.append({name: p}); sel[seed] = idx.tolist()
            np.save(OUT / f"caco2_{arm}_s{seed}_test.npy", p)
        res = group.evaluate_many(preds)[name]
        overlap = {s: len(set(sel[s]) & set(feature_candidates[s])) for s in SEEDS} if arm.startswith("inner") else None
        rep[arm] = {"mae": res, "seconds": round(time.time() - t), "overlap_with_released": overlap}
        print(f"[calici] {arm}: MAE {res[0]:.4f} +- {res[1]:.4f} ({time.time() - t:.0f}s)"
              + (f" | overlap with released subsets {overlap}" if overlap else ""), flush=True)
        (OUT / f"audit_{'_'.join(a.arms)}.json").write_text(json.dumps(rep, indent=1))
    print("CALICI_DONE", flush=True)


if __name__ == "__main__":
    main()
