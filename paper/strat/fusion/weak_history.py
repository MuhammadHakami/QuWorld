"""Can long-history context explain the entry's slowly varying errors? Shot-level 5-fold ridge (GPU) of the residuals
(PCA coefficients 0-6; v8 head residuals of R_axis, kappa, li) on per-frame inputs vs per-frame inputs + exponential
moving averages over the shot's frames (tau 0.1 / 0.3 / 1 / 3 s, causal and anti-causal) of the magnetics.

    .venv/bin/python paper/strat/fusion/weak_history.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import strategies as ST  # noqa: E402
from weak_resid import cv_ridge  # noqa: E402

TAUS = (100.0, 300.0, 1000.0, 3000.0)


def ema(x, t, tau, reverse=False):
    if reverse:
        return ema(x[::-1], -t[::-1], tau)[::-1]
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        a = 1 - np.exp(-(t[i] - t[i - 1]) / tau); out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    fn = d.fn; mag = [i for i, n in enumerate(fn) if n.startswith("mag_") and "lag" not in n]; ti = fn.index("t")
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    Xn, Xh, Y, G = [], [], [], []
    for s in shots:
        rows = fx.valid_rows(d, s); X = np.nan_to_num(np.asarray(d.X[rows], np.float64)); t = X[:, ti]
        hist = [ema(X[:, mag], t, tau, rev) for tau in TAUS for rev in (False, True)]
        Xn.append(np.c_[X, d.edge_feats()[rows]]); Xh.append(np.c_[X, d.edge_feats()[rows], *hist, t - t[0]])
        c = f(s)[0].mean(0); ct = np.asarray(d.C[rows][:, :7], np.float64)
        gt = pe[s]["cgt"]; ph = Ph[[pos[int(r)] for r in rows]]
        Y.append(np.c_[ct - c[:, :7], gt[:, [0, 2, 6]] - ph[:, [0, 2, 6]]]); G.append(np.full(len(rows), s))
    Y = np.concatenate(Y); ok = np.isfinite(Y).all(1)
    R = torch.tensor(Y[ok], device="cuda"); g = torch.tensor(np.concatenate(G)[ok], device="cuda")
    names = [f"c{k}" for k in range(7)] + ["h_Raxis", "h_kappa", "h_li"]
    for nmx, X in (("per-frame", Xn), ("+history", Xh)):
        X = torch.tensor(np.concatenate(X)[ok], device="cuda")
        for lam in (1e-3, 1e-2, 1e-1):
            Rh = cv_ridge(X, R, g, lam); r2 = 1 - ((R - Rh) ** 2).sum(0) / (R ** 2).sum(0)
            print(f"[history] {nmx:9s} lam {lam:g}: CV R2 of residual " + " ".join(f"{n}:{v:+.3f}" for n, v in zip(names, r2)), flush=True)


if __name__ == "__main__":
    main()
