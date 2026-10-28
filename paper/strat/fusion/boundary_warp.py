"""Doc 35 §37q: radial warp of the predicted psi so the scorer's LCFS lands on the shape heads' boundary (inference only).

Polar coordinates about the field's own O-point x0 (after the §37n shift). r_p(theta): the LCFS the scorer extracts from
psi, measured along N_ANG rays; r_t(theta): the target radii (heads). For a target point at radius rho, angle theta:
source radius = rho r_p / r_t inside (rho <= r_t), and rho - (r_t - r_p) exp(-((rho - r_t) / W)^2) outside, so the
psi_b level set moves from r_p to r_t and the far field is unchanged. Angles are interpolated periodically and linearly;
|r_t - r_p| is capped at CAP m. psi'(x) = psi(source(x)), bicubic.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

N_ANG, W, CAP = 32, 0.10, 0.05


def make_boundary_warp(R, Z, mask, sign, machine="DIII-D", n_iter=22):
    from lcfs import extract_lcfs_with_sign
    from o_point import find_o_point
    from boundary_labels import ray_radii
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    RR, ZZ = np.meshgrid(R, Z); mf = mask.astype(np.float64)

    def interp(r, th):                                          # ray count = len(r) (32 in §37q, 64 in §37s)
        ang = 2 * np.pi * np.arange(len(r)) / len(r)
        return np.interp(th, np.r_[ang, 2 * np.pi], np.r_[r, r[0]])

    def warp(psi, r_t):
        C, _ = extract_lcfs_with_sign(psi, R, Z, machine, mask, mf, n_iter=n_iter, axis_sign=sign)
        if C is None or len(C) < 8 or not np.isfinite(r_t).all():
            return psi, np.nan
        R0, Z0, *_ = find_o_point(sign * psi, R, Z, mask if mask.any() else None)
        r_p = ray_radii(np.asarray(C, float), R0, Z0, len(r_t))
        if not np.isfinite(r_p).all():
            return psi, np.nan
        r_t = r_p + np.clip(r_t - r_p, -CAP, CAP)
        dx, dy = RR - R0, ZZ - Z0; rho = np.hypot(dx, dy); th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        rp, rt = interp(r_p, th), interp(r_t, th)
        src = np.where(rho <= rt, rho * rp / rt, rho - (rt - rp) * np.exp(-((rho - rt) / W) ** 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(rho > 0, src / rho, 1.0)
        Rs, Zs = R0 + dx * f, Z0 + dy * f
        out = map_coordinates(psi, [(Zs - Z[0]) / dZ, (Rs - R[0]) / dR], order=3, mode="nearest")
        return out, float(np.sqrt(np.mean((r_t - r_p) ** 2)))
    return warp
