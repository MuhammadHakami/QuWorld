"""Doc 35 §37o: interior flux-profile warp so the scorer's li moves toward the model's li evidence (CPU, inference only).

Inside the LCFS the scorer extracts (cells inside the polygon with psi_N in [0, 1]), phi' = phi_ax - (phi_ax - phi_b) f(psi_N),
f(x) = x + beta x (1 - x)^2: f(0) = 0, f(1) = 1, f'(1) = 1, so the axis cell, the LCFS level and the boundary gradient
(Ampere's mu0 Ip in the scorer's li) are kept; only <Bp^2>_V moves. beta per frame by two secant steps on the scorer's own
internal_inductance, |beta| <= 0.5 (f stays monotone), target = (li head + li derived) / 2, or the head (full=True).
"""
from __future__ import annotations

import numpy as np

BETA_CAP = 0.5


def make_li_warp(R, Z, mask, sign, machine="DIII-D", n_iter=22):
    from derive import _points_in_polygon, _bilinear, internal_inductance
    from lcfs import extract_lcfs_with_sign
    RR, ZZ = np.meshgrid(R, Z)
    mf = mask.astype(np.float64)

    def warp(psi, li_head, full=False):
        C, _ = extract_lcfs_with_sign(psi, R, Z, machine, mask, mf, n_iter=n_iter, axis_sign=sign)
        if C is None or len(C) < 8:
            return psi, 0.0, np.nan
        phi = sign * psi
        inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
        if inside.sum() < 4:
            return psi, 0.0, np.nan
        phi_b = float(np.mean(_bilinear(phi, R, Z, C[:, 0], C[:, 1]))); phi_ax = float(phi[inside].max())
        if not phi_ax > phi_b:
            return psi, 0.0, np.nan
        x = (phi_ax - phi) / (phi_ax - phi_b); sel = inside & (x >= 0) & (x <= 1); xs = x[sel]

        def apply(beta):
            out = phi.copy(); out[sel] = phi_ax - (phi_ax - phi_b) * (xs + beta * xs * (1 - xs) ** 2)
            return sign * out
        li0 = internal_inductance(psi, R, Z, machine, C)
        if not np.isfinite(li0) or not np.isfinite(li_head):
            return psi, 0.0, li0
        tgt = li_head if full else 0.5 * (li0 + li_head)
        b1 = 0.1; li1 = internal_inductance(apply(b1), R, Z, machine, C)
        slope = (li1 - li0) / b1
        if not np.isfinite(slope) or slope == 0:
            return psi, 0.0, li0
        beta = float(np.clip((tgt - li0) / slope, -BETA_CAP, BETA_CAP))
        li2 = internal_inductance(apply(beta), R, Z, machine, C)
        if np.isfinite(li2) and beta != b1 and li2 != li1:
            beta = float(np.clip(beta + (tgt - li2) * (beta - b1) / (li2 - li1), -BETA_CAP, BETA_CAP))
        return apply(beta), beta, li0
    return warp
