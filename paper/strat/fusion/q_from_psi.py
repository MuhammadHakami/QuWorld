"""Fusion §41q: physics q95 from a psi map. q(psi_N) = (F / 2 pi) * loop integral dl / (R |grad psi|) on the psi_N = 0.95
surface, F ~ c * bcoil (vacuum toroidal field function; one global constant fitted on TRAIN-free VALID frames here only to
measure the ceiling). Axis / LCFS by the vendored scorer's own routines (axis = max of -psi on DIII-D). Ceiling test on TRUE
psi: R2 of c * |bcoil| * G(psi) vs EFIT q95.

    .venv/bin/python paper/strat/fusion/q_from_psi.py --n 2000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import fx  # noqa: F401,I001
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
import sota  # noqa: E402
from contour import encloses_point, find_contours, is_closed  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402
from o_point import find_o_point  # noqa: E402

G_ = np.load(fx.OUT / "efit_grid.npz"); RG, ZG = G_["R"].astype(np.float64), G_["Z"].astype(np.float64)


def bilinear(F, r, z):
    fr = np.clip((r - RG[0]) / (RG[1] - RG[0]), 0, 63.999); fz = np.clip((z - ZG[0]) / (ZG[1] - ZG[0]), 0, 63.999)
    i, j = fr.astype(int), fz.astype(int); a, b = fr - i, fz - j
    return (1 - a) * (1 - b) * F[j, i] + a * (1 - b) * F[j, i + 1] + (1 - a) * b * F[j + 1, i] + a * b * F[j + 1, i + 1]


def G_of(psi, level_n=0.95):
    """(1/2pi) * loop integral dl / (R |grad psi|) on the psi_N = level_n surface; nan on failure."""
    phi = -np.asarray(psi, np.float64)                              # DIII-D: axis = max of -psi
    _, _, iz, ir, _ = find_o_point(phi, RG, ZG, None)
    C = extract_lcfs(psi, RG, ZG, "DIII-D")
    if C is None:
        return np.nan
    pa = float(phi[iz, ir]); pb = float(np.mean(bilinear(phi, C[:, 0], C[:, 1])))
    lev = pa - level_n * (pa - pb)
    for c in find_contours(phi, lev):
        if len(c) >= 8 and is_closed(c) and encloses_point(c, iz, ir):
            z = np.interp(c[:, 0], np.arange(65), ZG); r = np.interp(c[:, 1], np.arange(65), RG)
            gz, gr = np.gradient(phi, ZG, RG); gp = np.hypot(bilinear(gr, r, z), bilinear(gz, r, z))
            rm, zm, gm = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1]), 0.5 * (gp[1:] + gp[:-1])
            dl = np.hypot(np.diff(r), np.diff(z))
            return float(np.sum(dl / (rm * gm)) / (2 * np.pi))
    return np.nan


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); a = ap.parse_args()
    d = sota.Data(); va = np.nonzero(d.split == 1)[0]; rows = np.sort(np.random.default_rng(0).choice(va, a.n, replace=False))
    import inputs_fix
    inputs_fix.use(d, name="X_ipfix"); bc = np.abs(d.inputs(rows, np.array([d.fn.index("mag_bcoil")]))[:, 0]).astype(np.float64)
    q = d.S[rows, sota.LABELS.index("efit_q95")].astype(np.float64); P = d.psi_of(rows).reshape(-1, 65, 65)
    G = np.array([G_of(p) for p in P]); ok = np.isfinite(G) & np.isfinite(q) & (bc > 1)
    x = bc[ok] * G[ok]; c = float(np.sum(x * q[ok]) / np.sum(x * x)); pred = c * x
    r2 = 1 - np.mean((q[ok] - pred) ** 2) / np.var(q[ok])
    lr = np.polyfit(np.log(x), np.log(q[ok]), 1); r2l = 1 - np.mean((q[ok] - np.exp(np.polyval(lr, np.log(x)))) ** 2) / np.var(q[ok])
    print(f"[qpsi] frames {ok.sum()}/{len(rows)} c {c:.5g}: R2 {r2:.5f} | log-linear fit {lr.round(4).tolist()} R2 {r2l:.5f} | "
          f"median rel err {np.median(np.abs(pred / q[ok] - 1)):.4f}", flush=True)
    np.savez(fx.OUT / "q_from_psi_true.npz", rows=rows, G=G, bc=bc, q=q)


if __name__ == "__main__":
    main()
