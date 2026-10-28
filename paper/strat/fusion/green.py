"""Doc 35 §41e: Green's-function (vacuum) flux of the DIII-D coils on the EFIT grid - the exactly linear part of psi.

psi(R, Z) [Wb/rad] of a circular filament at (a, b) carrying I [A]: psi = mu0 I sqrt(a R) / (pi k) [(1 - k^2 / 2) K(k^2) - E(k^2)],
k^2 = 4 a R / ((a + R)^2 + (Z - b)^2). Each coil (19 lumped elements from the shot's machine geometry: 18 F-coils + the E-coil
solenoid; rectangles or parallelograms with angle1 / angle2 as in EFIT mhdin) is split into n x n sub-filaments sharing its
ampere-turns. Output: <cache>/green.npz with G (19, 65 * 65) per unit current, the coil -> input-signal map and the grid.

    .venv/bin/python paper/strat/fusion/green.py
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy.special import ellipe, ellipk

import fx  # noqa: I001
import sys
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

MU0 = 4e-7 * np.pi


def filament_psi(a, b, R, Z):
    m = np.clip(4 * a * R / ((a + R) ** 2 + (Z - b) ** 2), 1e-12, 1 - 1e-12); k = np.sqrt(m)
    return MU0 * np.sqrt(a * R) / (np.pi * k) * ((1 - m / 2) * ellipk(m) - ellipe(m))


def coil_points(R, Z, w, h, a1, a2, n=12):
    """Sub-filament centres of an EFIT coil element: a w x h rectangle, sheared by angle1 (top edge shifted in R) or
    angle2 (side edges shifted in Z), as in EFIT's parallelogram convention."""
    u = (np.arange(n) + 0.5) / n - 0.5; du, dv = np.meshgrid(u, u, indexing="ij"); x, y = du * w, dv * h
    if a1:
        x = x + y / np.tan(np.radians(a1)) if abs(a1) != 90 else x
    if a2:
        y = y + x * np.tan(np.radians(a2)) if abs(a2) != 90 else y
    return (R + x).ravel(), (Z + y).ravel()


def main() -> None:
    f = sorted(glob.glob(str(fx.ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data" / "diii_d_train" / "*.parquet")))[0]
    r = pd.read_parquet(f).iloc[0]
    gR, gZ = np.asarray(r["efit_grid_R"], np.float64), np.asarray(r["efit_grid_Z"], np.float64)
    RR, ZZ = np.meshgrid(gR, gZ, indexing="xy")                              # psirz layout (Z rows, R columns)
    G = []
    for i in range(len(r["coil_R"])):
        pr, pz = coil_points(r["coil_R"][i], r["coil_Z"][i], r["coil_width"][i], r["coil_height"][i], r["coil_angle1"][i], r["coil_angle2"][i])
        G.append(sum(filament_psi(a, b, RR, ZZ) for a, b in zip(pr, pz)).ravel() / len(pr))
    G = np.stack(G); names = [str(c) for c in r["coil_name"]]; cols = [str(c) for c in r["coil_input_column"]]
    np.savez(sota.CACHE / "green.npz", G=G, names=names, cols=cols, R=gR, Z=gZ)
    print(f"[green] {G.shape}; coils {list(zip(names, cols))}", flush=True)


if __name__ == "__main__":
    main()
