"""Doc 35 §38w-pre: are the heads' per-shot offsets learnable from whole-shot dynamics? (gate for sequence heads, doc 36)

§38t/§38u: shot-mean-pooled inputs (+ head outputs) do not predict the out-of-fold shot offsets. A sequence model only
pays if the offset is identifiable from how the inputs move within the shot. Cheap test: per TRAIN shot, summary
statistics of every base input over its frames (mean, std, p10, p90, end-minus-start slope, frame count) -> GBDT,
5-fold by shot -> out-of-fold R2 of the standardized shot-mean residual of each consistency scalar (§38u OOF heads).
Reference arm: the same GBDT on the means only (what §38t/u already had). Gate: dynamics R2 >= 0.10 on at least two of
kappa, li, R_axis and above the means-only arm.

    .venv/bin/python paper/strat/fusion/offset_learnability.py   -> runs/strat/fusion/offset_learnability.json
"""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import fx


def shot_residuals():
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); names7 = [str(x) for x in lab["names"]]
    pos = {r: i for i, r in enumerate(lab["rows_train"])}
    rows, pred = [], []
    for k in range(5):
        z = np.load(fx.OUT / f"head_boundary_v8_J1F{k}_s1_oof.npz"); rows.append(z["rows_oof"]); pred.append(z["pred_oof"][:, :7])
    rows, pred = np.concatenate(rows), np.concatenate(pred)
    y = lab["y_train"][[pos[r] for r in rows], :7]; sd = np.nanstd(lab["y_train"][:, :7], 0)
    shot = np.load(fx.ROOT / "data/sophelio_proc/cache_k256/shot.npy")[rows]
    us, inv = np.unique(shot, return_inverse=True); r = (pred - y) / sd
    ok = np.isfinite(r); res = np.zeros((len(us), 7)); cnt = np.zeros((len(us), 7))
    np.add.at(res, inv, np.where(ok, r, 0)); np.add.at(cnt, inv, ok)
    return names7, us, res / np.maximum(cnt, 1), rows, shot


def features(rows: np.ndarray, shot: np.ndarray, us: np.ndarray):
    sota, d = fx.data(); fn = list(d.fn); base = [i for i, n in enumerate(fn) if not re.search(r"_lag[-+]\d+ms$", n)]
    order = np.lexsort((rows, shot)); rows, shot = rows[order], shot[order]
    X = np.asarray(d.X[rows][:, base], np.float64); bounds = np.searchsorted(shot, us)
    bounds = np.append(bounds, len(shot)); F, M = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        x = X[a:b]; q = max(1, len(x) // 10)
        F.append(np.concatenate([x.mean(0), x.std(0), np.percentile(x, 10, 0), np.percentile(x, 90, 0),
                                 x[-q:].mean(0) - x[:q].mean(0), [len(x)]]))
        M.append(x.mean(0))
    return np.array(F), np.array(M)


def oof_r2(F: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    p = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(F, y, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15, random_state=0)
        p[te] = m.fit(F[tr], y[tr]).predict(F[te])
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    names7, us, res, rows, shot = shot_residuals()
    F, M = features(rows, shot, us); g = np.arange(len(us))
    rep = dict(shots=len(us), n_features=F.shape[1], scalars={})
    for j, n in enumerate(names7):
        y = res[:, j]
        r_dyn, r_mean = oof_r2(F, y, g), oof_r2(M, y, g)
        rep["scalars"][n] = dict(dynamics_r2=r_dyn, means_r2=r_mean, offset_rms_sd=float(np.sqrt((y ** 2).mean())))
        print(f"[offset-learn] {n:8s} offset rms {np.sqrt((y ** 2).mean()):.3f} sd | OOF R2 dynamics {r_dyn:+.3f} | means only {r_mean:+.3f}", flush=True)
    (fx.OUT / "offset_learnability.json").write_text(json.dumps(rep, indent=1))
    print("OFFSET_LEARN_DONE", flush=True)


if __name__ == "__main__":
    main()
