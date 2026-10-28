"""Fusion §45o: vertical-equilibrium inputs. In vertical force balance the plasma sits where the external radial field on its current
vanishes, so the height of the vacuum B_R null is a physical estimate of Z_axis that extrapolates to unusual coil patterns (where our heads
fail: CV OOD shots Z_axis R2 0.941 vs 0.993, §45n). The §45j vac block gave the network B_R and B_Z at three points; the null height is a
ratio of those, which an MLP cannot form outside the training range. Per frame, from the 18 measured F-coil currents (kA-turn) through
the shipped coil geometry (fb/geom.py Green's functions; the E-coil and vessel are unmeasured and left out):
  on vertical lines R = 1.50 / 1.70 / 1.90 m, Z in [-0.6, 0.6] (1 cm): B_R(Z) = -(1/R) dpsi/dZ;
  z_null = the zero of B_R nearest Z = 0 (clipped to +-0.6; +-0.6 when none), z_lin = -B_R(0) / dB_R/dZ(0) (clipped to +-0.6),
  n_lin = dB_R/dZ(0) x 2 pi R / (mu0 |Ip|) (field-curvature index), b_r0 = B_R(0) x 2 pi R / (mu0 |Ip|)       -> 3 x 4 = 12 inputs.
    import vz_inputs as VZ; VZ.use(data)        # wrap after the other input blocks
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fb"))
MU0 = 4e-7 * np.pi
COILS = [f"F{i}{s}" for i in range(1, 10) for s in "AB"]
RL = (1.50, 1.70, 1.90)
ZL = np.round(np.arange(-0.60, 0.6001, 0.01), 4)
_T = {}


def tables():
    """psi per kA-turn of each coil at (R, Z +- h) on the three vertical lines -> dpsi/dZ tables (18, 3, nZ)."""
    if "D" in _T:
        return _T
    import geom
    r = geom.first_row("d3d"); names = [str(c) for c in r["coil_name"]]; h = 0.002
    D = np.zeros((len(COILS), len(RL), len(ZL)))
    for k, c in enumerate(COILS):
        i = names.index(c)
        pr, pz = geom.coil_points(r["coil_R"][i], r["coil_Z"][i], r["coil_width"][i], r["coil_height"][i], r["coil_angle1"][i], r["coil_angle2"][i], 6)
        for a, R in enumerate(RL):
            up = sum(geom.filament_psi(x, y, np.full(len(ZL), R), ZL + h) for x, y in zip(pr, pz)) / len(pr)
            dn = sum(geom.filament_psi(x, y, np.full(len(ZL), R), ZL - h) for x, y in zip(pr, pz)) / len(pr)
            D[k, a] = (up - dn) / (2 * h) * 1e3                                                   # dpsi/dZ per kA-turn
    _T.update(D=D)
    return _T


def feats(x: np.ndarray, cols: list[str]) -> np.ndarray:
    T = tables(); ix = {n: i for i, n in enumerate(cols)}
    I = np.stack([np.asarray(x[:, ix[f"mag_{c}"]], np.float64) for c in COILS], 1)                # kA-turn
    ip = np.abs(np.asarray(x[:, ix["mag_plasma_current"]], np.float64)) + 100.0                    # kA floor keeps ramps bounded
    out, z0 = [], int(np.argmin(np.abs(ZL)))
    for a, R in enumerate(RL):
        BR = -(I @ T["D"][:, a]) / R                                                              # (n, nZ) tesla
        sgn = np.sign(BR); cross = (sgn[:, :-1] * sgn[:, 1:]) <= 0                                # sign change between Z_k, Z_k+1
        zc = ZL[:-1] + 0.01 * np.abs(BR[:, :-1]) / np.maximum(np.abs(BR[:, :-1]) + np.abs(BR[:, 1:]), 1e-12)
        dist = np.where(cross, np.abs(zc), np.inf); j = dist.argmin(1); has = np.isfinite(dist[np.arange(len(BR)), j])
        zn = np.where(has, zc[np.arange(len(BR)), j], np.where(BR[:, z0] > 0, -0.6, 0.6))
        g = (BR[:, z0 + 1] - BR[:, z0 - 1]) / 0.02
        zl = np.clip(-BR[:, z0] / np.where(np.abs(g) > 1e-9, g, 1e-9), -0.6, 0.6)
        norm = 2 * np.pi * R / (MU0 * ip * 1e3)
        out += [zn, zl, g * norm, BR[:, z0] * norm]
    F = np.stack(out, 1)
    return np.clip(np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3).astype(np.float32)


def use(data):
    inner, fn = data.inputs, list(data.fn)

    def inputs(rows, fm, edge=False, hist=False, thz=False):
        x = inner(rows, fm, edge, hist=hist, thz=thz) if (hist or thz) else inner(rows, fm, edge)
        cols = [n for n, m in zip(fn, fm) if m]; base = np.asarray(x[:, :len(cols)])
        return np.concatenate([np.asarray(x, np.float32), feats(base, cols)], 1)
    data.inputs = inputs
    return data
