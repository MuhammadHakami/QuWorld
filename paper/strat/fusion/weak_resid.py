"""Is the entry's psi-coefficient residual predictable from the inputs? Shot-level 5-fold ridge on the VALID sample
(GPU). A high CV R^2 means the information is in the inputs and the trunk under-uses it (capacity / optimisation);
a low one means it is missing. Feature groups: magnetics now, + lags, + Thomson, + time.

    .venv/bin/python paper/strat/fusion/weak_resid.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import strategies as ST  # noqa: E402

KS = (0, 1, 2, 4, 5, 6)


def cv_ridge(X, Y, g, lam=1.0, k=5):
    out = torch.zeros_like(Y); u = torch.unique(g); perm = u[torch.randperm(len(u), generator=torch.Generator().manual_seed(0))]
    for i in range(k):
        te = torch.isin(g, perm[i::k]); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-8); Xt = (X - mu) / sd
        Xt = torch.cat([Xt, torch.ones(len(Xt), 1, device=X.device, dtype=X.dtype)], 1)
        A = Xt[tr].T @ Xt[tr] + lam * len(Xt) * torch.eye(Xt.shape[1], device=X.device, dtype=X.dtype)
        out[te] = Xt[te] @ torch.linalg.solve(A, Xt[tr].T @ Y[tr])
    return out


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); f, comps, mean, H = ST.entry_preds(d, sota)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots])
    P = np.concatenate([f(s)[0].mean(0) for s in shots]); Ct = np.asarray(d.C[rows][:, :P.shape[1]], np.float64)
    dev = "cuda"; R = torch.tensor(Ct - P, device=dev); g = torch.tensor(d.shot[rows], device=dev)
    X = np.asarray(d.X[rows], np.float64); X = np.where(np.isfinite(X), X, 0.0); fn = d.fn
    groups = {"mag_now": [i for i, n in enumerate(fn) if n.startswith("mag_") and "lag" not in n],
              "mag_all": [i for i, n in enumerate(fn) if n.startswith("mag_")],
              "all": list(range(len(fn)))}
    edge = d.edge_feats()[rows]
    tot = (R ** 2).sum()
    for name, idx in groups.items():
        Xg = torch.tensor(np.c_[X[:, idx], edge, P[:, :8]], device=dev)
        for lam in (1e-4, 1e-3, 1e-2):
            Rh = cv_ridge(Xg, R, g, lam)
            r2 = 1 - ((R - Rh) ** 2).sum(0) / (R ** 2).sum(0)
            print(f"[resid] {name:8s} lam {lam:g}: total psi sq-err removed {1 - ((R - Rh) ** 2).sum() / tot:+.3f}; "
                  f"per-coef CV R2 " + " ".join(f"c{k}:{r2[k]:+.3f}" for k in KS), flush=True)


if __name__ == "__main__":
    main()
