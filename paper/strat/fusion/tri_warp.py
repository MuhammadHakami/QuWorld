"""Doc 35 §37x: horizontal shear of the upper / lower LCFS region so the scorer's tri_top / tri_bot hit the heads.

tri = (Rgeo - R at the highest (lowest) point) / a. In target space the source is R - d_top f((Z - Zg)/(Zmax - Zg)) w
above the geometric centre and R - d_bot f((Zg - Z)/(Zg - Zmin)) w below it, with f(u) = clip(u, 0, 1.5)^2 (zero at
the midplane, so Rmax / Rmin and a barely move) and w = 1 inside the current LCFS, exp(-(d/W)^2) outside. Moving the
top point by d changes tri_top by -d/a. d_top and d_bot come from two secant steps each on the scorer's own
extraction; |d| <= CAP. Guard: if |tri_top - t_top| + |tri_bot - t_bot| ends larger than before, the frame is left
unwarped.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

W, CAP, N_RAYS = 0.10, 0.08, 64


def make_tri_warp(R, Z, mask, sign, machine="DIII-D", n_iter=22):
    from lcfs import extract_lcfs_with_sign
    from derive import _shape_from_contour, _points_in_polygon
    from boundary_labels import ray_radii
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    RR, ZZ = np.meshgrid(R, Z); mf = mask.astype(np.float64)
    ang = 2 * np.pi * np.arange(N_RAYS) / N_RAYS

    def shape(psi):
        C, _ = extract_lcfs_with_sign(psi, R, Z, machine, mask, mf, n_iter=n_iter, axis_sign=sign)
        if C is None or len(C) < 8:
            return None, None
        C = np.asarray(C, float)
        return C, _shape_from_contour(C)

    def warp(psi, t_top, t_bot):
        C, s0 = shape(psi)
        if C is None or not all(np.isfinite([s0["tri_top"], s0["tri_bot"], t_top, t_bot])):
            return psi, (0.0, 0.0)
        Rmax, Rmin, Zmax, Zmin = C[:, 0].max(), C[:, 0].min(), C[:, 1].max(), C[:, 1].min()
        Rg, Zg, a = 0.5 * (Rmax + Rmin), 0.5 * (Zmax + Zmin), 0.5 * (Rmax - Rmin)
        rp = ray_radii(C, Rg, Zg, N_RAYS)
        if not np.isfinite(rp).all():
            return psi, (0.0, 0.0)
        dx, dy = RR - Rg, ZZ - Zg; rho = np.hypot(dx, dy); th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        rpt = np.interp(th, np.r_[ang, 2 * np.pi], np.r_[rp, rp[0]])
        inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
        w = np.where(inside | (rho <= rpt), 1.0, np.exp(-((rho - rpt) / W) ** 2))
        fu = np.clip((ZZ - Zg) / (Zmax - Zg), 0, 1.5) ** 2; fl = np.clip((Zg - ZZ) / (Zg - Zmin), 0, 1.5) ** 2

        def apply(d_top, d_bot):
            Rs = RR - (d_top * fu + d_bot * fl) * w
            return map_coordinates(psi, [(ZZ - Z[0]) / dZ, (Rs - R[0]) / dR], order=3, mode="nearest")

        def miss(s):
            return abs(s["tri_top"] - t_top) + abs(s["tri_bot"] - t_bot)
        d = np.clip(np.array([-(t_top - s0["tri_top"]) * a, -(t_bot - s0["tri_bot"]) * a]), -CAP, CAP)
        out = apply(*d)
        for _ in range(2):
            _, s1 = shape(out)
            if s1 is None or not np.isfinite([s1["tri_top"], s1["tri_bot"]]).all():
                break
            d = np.clip(d - np.array([(t_top - s1["tri_top"]) * a, (t_bot - s1["tri_bot"]) * a]), -CAP, CAP)
            out = apply(*d)
        _, sf = shape(out)
        if sf is None or not np.isfinite([sf["tri_top"], sf["tri_bot"]]).all() or miss(sf) > miss(s0):
            return psi, (0.0, 0.0)
        return out, (float(d[0]), float(d[1]))
    return warp
