"""Doc 35 §46 validation helper: EFIT's own plasma current and external flux for labelled DIII-D frames (physical sign).
J = -Delta* psi / (mu0 R) restricted to the EFIT plasma (same sign as the net current, > 2% of its peak, off the grid edge);
psi_ext = psi - Gpp @ (J dA). Lets the solver be tested with a perfect external field, isolating the algorithm from the coil set.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


def efit_split(psi_phys: np.ndarray, m) -> tuple:
    R, Z = m.R, m.Z; RR = np.meshgrid(R, Z, indexing="xy")[0]; dA = (R[1] - R[0]) * (Z[1] - Z[0])
    dpdR = np.gradient(psi_phys, R, axis=2); d2Z = np.gradient(np.gradient(psi_phys, Z, axis=1), Z, axis=1)
    J = -(RR * np.gradient(dpdR / RR, R, axis=2) + d2Z) / (MU0 * RR)
    s = np.sign(J.sum((1, 2)))[:, None, None]
    Jm = np.where((np.sign(J) == s) & (np.abs(J) > 0.02 * np.abs(J).max((1, 2), keepdims=True)), J, 0.0)
    Jm[:, :2] = 0; Jm[:, -2:] = 0; Jm[:, :, :2] = 0; Jm[:, :, -2:] = 0
    pp = (Jm.reshape(len(Jm), -1) * dA) @ m.Gpp
    return Jm, psi_phys - pp.reshape(psi_phys.shape), Jm.sum((1, 2)) * dA
