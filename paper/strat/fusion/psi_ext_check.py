"""Doc 35 §42w feasibility: split the true psi into the plasma part (Green's function of the grid toroidal current J = -D*psi / (mu0 R))
and the external part psi_ext = psi - psi_pl (coils + vessel: linear in their currents). Checks: grid current vs Ip, variance split,
and how well a LINEAR model of the magnetics inputs predicts psi, psi_ext, psi_pl on held-out VT shots (fit on half the shots).
    .venv/bin/python paper/strat/fusion/psi_ext_check.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from scipy.special import ellipe, ellipk

import fx

MU0 = 4e-7 * np.pi


def green(R, Z):
    """(4225, 4225) flux per radian at grid points from a unit ring current at each grid point; self term = 2x2 sub-cell average."""
    RR, ZZ = np.meshgrid(R, Z); r, z = RR.ravel(), ZZ.ravel(); dR, dZ = R[1] - R[0], Z[1] - Z[0]

    def g(r1, z1, r2, z2):
        m = 4 * r1 * r2 / ((r1 + r2) ** 2 + (z1 - z2) ** 2); m = np.minimum(m, 1 - 1e-12); k = np.sqrt(m)
        return MU0 / (2 * np.pi) * np.sqrt(r1 * r2) * ((2 - m) * ellipk(m) - 2 * ellipe(m)) / k
    G = g(r[:, None], z[:, None], r[None, :], z[None, :])
    s = [g(r, z, r + a * dR / 4, z + b * dZ / 4) for a in (-1, 1) for b in (-1, 1)]
    G[np.arange(len(r)), np.arange(len(r))] = np.mean(s, 0)
    return G


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    import axis_shift as AX
    d = sota.Data(); R, Z, _ = AX.grid(); dR, dZ = R[1] - R[0], Z[1] - Z[0]; dev = "cuda"
    G = torch.tensor(green(R, Z), device=dev); Rg = torch.tensor(R, device=dev)
    vt = np.unique(d.shot[d.split != 0]); rng = np.random.default_rng(0); sh = rng.choice(vt, 300, replace=False)
    rows = np.concatenate([np.nonzero(d.shot == s)[0][::2] for s in sh]); print(f"[ext] {len(sh)} shots, {len(rows)} frames", flush=True)
    fn = d.fn; mag = [i for i, n in enumerate(fn) if n.startswith("mag_")]; ip = fn.index("mag_plasma_current")
    P, E, L, Igrid = [], [], [], []
    for i in range(0, len(rows), 2048):
        rr = rows[i:i + 2048]; p = torch.tensor(np.asarray(d.psi_of(rr), np.float64).reshape(-1, 65, 65), device=dev)
        d2R = (p[:, 1:-1, 2:] - 2 * p[:, 1:-1, 1:-1] + p[:, 1:-1, :-2]) / dR ** 2
        d1R = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2 * dR); d2Z = (p[:, 2:, 1:-1] - 2 * p[:, 1:-1, 1:-1] + p[:, :-2, 1:-1]) / dZ ** 2
        ds = d2R - d1R / Rg[1:-1] + d2Z; J = torch.zeros_like(p); J[:, 1:-1, 1:-1] = -ds / (MU0 * Rg[1:-1])
        Igrid.append((J.sum((1, 2)) * dR * dZ).cpu().numpy())
        pl = (J.reshape(len(rr), -1) * dR * dZ) @ G.T
        P.append(p.reshape(len(rr), -1).cpu().numpy()); L.append(pl.cpu().numpy()); E.append(P[-1] - L[-1])
    P, L, E, Igrid = np.concatenate(P), np.concatenate(L), np.concatenate(E), np.concatenate(Igrid)
    Ip = np.asarray(d.X[rows, ip], np.float64)
    print(f"[ext] grid current / Ip: median {np.median(Igrid / Ip):.4f}, p10 {np.quantile(Igrid / Ip, .1):.4f}, p90 {np.quantile(Igrid / Ip, .9):.4f}; "
          f"|Ip| median {np.median(np.abs(Ip)):.3g}", flush=True)
    tv = lambda A: ((A - A.mean(0)) ** 2).sum()
    print(f"[ext] variance share: psi_ext {tv(E) / tv(P):.4f}, psi_pl {tv(L) / tv(P):.4f}", flush=True)
    X = np.asarray(d.X[rows][:, mag], np.float64); X = np.c_[X, np.ones(len(X))]
    tr = np.isin(d.shot[rows], sh[:150]); te = ~tr
    for nm, Y in (("psi", P), ("psi_ext", E), ("psi_pl", L)):
        W, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None); e = Y[te] - X[te] @ W
        print(f"[ext] linear(magnetics, {X.shape[1] - 1} feats) -> {nm}: held-out R2 {1 - (e ** 2).sum() / tv(Y[te]):.6f}, rms {np.sqrt((e ** 2).mean()):.4g}"
              f" (of psi sd {P.std():.4g})", flush=True)
    np.savez(fx.OUT / "psi_ext_check.npz", rows=rows, Igrid=Igrid)
    print("EXT_DONE", flush=True)


if __name__ == "__main__":
    main()
