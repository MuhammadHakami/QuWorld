"""Doc 35 §46 algorithm check on labelled DIII-D frames: solver driven by EFIT's own external flux and plasma-current integral (both
from the label, see ext_efit.py) - isolates profile / boundary / vertical-control quality from the coil-set question. Not used for MAST.
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python paper/strat/fusion/fb/test_ext.py [n_shots] [b0 am an] [vertical 0/1]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import geom  # noqa: E402
import data  # noqa: E402
from ext_efit import efit_split  # noqa: E402
from solve import Solver  # noqa: E402

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def r2(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    return 1 - np.sum((p[ok] - y[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    b0, am, an = (float(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (0.5, 1.0, 2.0)))
    vert = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
    m = geom.build("d3d"); fs = data.files("diii_d_train"); rng = np.random.default_rng(0); pick = sorted(rng.choice(len(fs), n, replace=False))
    D = data.load("diii_d_train", m.circuits, "d3d", [fs[i] for i in pick], labels=True)
    Y = D["psi"].astype(np.float64); J, ext, ipi = efit_split(Y, m)
    sv = Solver(m); t = time.time(); P = []
    for a in range(0, len(Y), 512):
        o = sv.solve(None, ipi[a:a + 512] / 1e3, b0=b0, am=am, an=an, R_init=1.7, pc=ext[a:a + 512], vertical=vert)
        P.append(o["psi"].cpu().numpy())
        print(f"[ext] batch {a}: iters {o['n_iter']} dJ med {float(o['dJ'].median()):.2e} p90 {float(o['dJ'].quantile(0.9)):.2e} | {time.time() - t:.0f}s", flush=True)
    P = np.concatenate(P).astype(np.float64)
    print(f"[ext] psi R2 pooled {r2(P.ravel(), Y.ravel()):.5f}  ext-only {r2(ext.ravel(), Y.ravel()):.5f}", flush=True)
    import axis_shift as AX
    from gpu_scorer import GPUScorer
    gR, gZ, mask = AX.grid(); sc = GPUScorer(gR, gZ, mask); dp, dy = [], []
    for a in range(0, len(P), 1024):
        op = sc.derive(torch.tensor(P[a:a + 1024])); oy = sc.derive(torch.tensor(Y[a:a + 1024]))
        dp.append(np.stack([op[k].cpu().numpy() for k in SC], 1)); dy.append(np.stack([oy[k].cpu().numpy() for k in SC], 1))
    dp, dy = np.concatenate(dp), np.concatenate(dy)
    print("[ext] consistency R2: " + " ".join(f"{k} {r2(dp[:, j], dy[:, j]):.4f}" for j, k in enumerate(SC)), flush=True)
    print("[ext] mean err: " + " ".join(f"{k} {np.nanmean(dp[:, j] - dy[:, j]):+.4f}" for j, k in enumerate(SC)), flush=True)


if __name__ == "__main__":
    main()
