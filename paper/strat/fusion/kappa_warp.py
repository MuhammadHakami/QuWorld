"""Doc 35 §37w: area-preserving elongation of the predicted LCFS region so the scorer's kappa hits the kappa head.

About the LCFS's geometric centre (Rg, Zg) (mid-points of its R and Z extents): T(R, Z) = (Rg + (R - Rg)/sqrt(s),
Zg + (Z - Zg) sqrt(s)) multiplies kappa by s and keeps the area. It keeps triangularity, since (Rg - R_top) and the
minor radius scale together. psi'(x) = psi(x + (T^-1(x) - x) w(x)), with w = 1 inside the current LCFS and
exp(-(d/W)^2) outside (d = radial distance beyond the LCFS along rays from (Rg, Zg)), so the far field is kept.
s by two secant steps on the scorer's own kappa; s is clipped to [1 - CAP, 1 + CAP]. Guard: if the final kappa is
farther from the target than before (the extraction jumped), the frame is left unwarped.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

W, CAP, N_RAYS = 0.10, 0.10, 64


def make_kappa_warp(R, Z, mask, sign, machine="DIII-D", n_iter=22):
    from lcfs import extract_lcfs_with_sign
    from derive import _shape_from_contour, _points_in_polygon
    from boundary_labels import ray_radii
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    RR, ZZ = np.meshgrid(R, Z); mf = mask.astype(np.float64)
    ang = 2 * np.pi * np.arange(N_RAYS) / N_RAYS

    def lcfs(psi):
        C, _ = extract_lcfs_with_sign(psi, R, Z, machine, mask, mf, n_iter=n_iter, axis_sign=sign)
        return None if C is None or len(C) < 8 else np.asarray(C, float)

    def kappa_of(C):
        return float(_shape_from_contour(C)["kappa"]) if C is not None else np.nan

    def warp(psi, kappa_t):
        C = lcfs(psi); k0 = kappa_of(C)
        if not (np.isfinite(k0) and np.isfinite(kappa_t)):
            return psi, 1.0
        Rg, Zg = 0.5 * (C[:, 0].max() + C[:, 0].min()), 0.5 * (C[:, 1].max() + C[:, 1].min())
        rp = ray_radii(C, Rg, Zg, N_RAYS)
        if not np.isfinite(rp).all():
            return psi, 1.0
        dx, dy = RR - Rg, ZZ - Zg; rho = np.hypot(dx, dy); th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        rpt = np.interp(th, np.r_[ang, 2 * np.pi], np.r_[rp, rp[0]])
        inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
        w = np.where(inside | (rho <= rpt), 1.0, np.exp(-((rho - rpt) / W) ** 2))

        def apply(s):
            Rs = RR + ((Rg + dx * np.sqrt(s)) - RR) * w
            Zs = ZZ + ((Zg + dy / np.sqrt(s)) - ZZ) * w
            return map_coordinates(psi, [(Zs - Z[0]) / dZ, (Rs - R[0]) / dR], order=3, mode="nearest")
        s = float(np.clip(kappa_t / k0, 1 - CAP, 1 + CAP)); out = apply(s)
        for _ in range(2):
            k1 = kappa_of(lcfs(out))
            if not np.isfinite(k1) or abs(k1 - kappa_t) < 1e-5:
                break
            s = float(np.clip(s * kappa_t / k1, 1 - CAP, 1 + CAP)); out = apply(s)
        kf = kappa_of(lcfs(out))
        if not np.isfinite(kf) or abs(kf - kappa_t) > abs(k0 - kappa_t):   # guard: extraction jumped (X-point flip)
            return psi, 1.0
        return out, s
    return warp
