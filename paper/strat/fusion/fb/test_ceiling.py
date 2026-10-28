"""Doc 35 §46b ceiling check on labelled DIII-D frames: the solver constrained by what MAST's midplane Thomson can supply - outer LCFS
radius (-> vertical-field knob) and magnetic-axis radius (-> pressure-profile knob b0) - here taken from EFIT (oracle) to bound what a
Thomson-constrained MAST solve could reach. External field: EFIT's own (ext) or the measured coils. Validation only (DIII-D labels).
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python paper/strat/fusion/fb/test_ceiling.py [n_shots] [am an] [n_iter]
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
from test_rout import efit_rout, r2, SC  # noqa: E402


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    am, an = (float(x) for x in (sys.argv[2:4] if len(sys.argv) > 3 else (2.0, 1.0)))
    nit = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    m = geom.build("d3d"); fs = data.files("diii_d_train"); rng = np.random.default_rng(0); pick = sorted(rng.choice(len(fs), n, replace=False))
    D = data.load("diii_d_train", m.circuits, "d3d", [fs[i] for i in pick], labels=True)
    Y = D["psi"].astype(np.float64); J, ext, ipi = efit_split(Y, m); sv = Solver(m)
    ro = np.concatenate([efit_rout(sv, Y[a:a + 512], ipi[a:a + 512]) for a in range(0, len(Y), 512)])
    import axis_shift as AX
    from gpu_scorer import GPUScorer
    gR, gZ, mask = AX.grid(); sc = GPUScorer(gR, gZ, mask)
    dy = np.concatenate([np.stack([v.cpu().numpy() for v in [sc.derive(torch.tensor(Y[a:a + 1024]))[k] for k in SC]], 1) for a in range(0, len(Y), 1024)])
    rax = dy[:, 0]
    s_ip = -np.sign(D["Ip"]); s_ip[s_ip == 0] = 1
    pcoil = s_ip[:, None, None] * sv.coil_psi(torch.tensor(np.nan_to_num(D["I"]))).cpu().numpy()
    for nm, pc in (("efit-ext", ext), ("coils", pcoil)):
        t = time.time(); P = []; b0s = []
        for a in range(0, len(Y), 512):
            o = sv.solve(None, ipi[a:a + 512] / 1e3, b0=0.5, am=am, an=an, R_init=1.7, pc=pc[a:a + 512], R_out=ro[a:a + 512], R_ax=rax[a:a + 512],
                         n_iter=nit, w=0.2, kp=0.15, ki=0.01)
            P.append(o["psi"].cpu().numpy()); b0s.append(o["b0"].cpu().numpy())
        P = np.concatenate(P).astype(np.float64); b0s = np.concatenate(b0s)
        dp = np.concatenate([np.stack([v.cpu().numpy() for v in [sc.derive(torch.tensor(P[a:a + 1024]))[k] for k in SC]], 1) for a in range(0, len(P), 1024)])
        c = [max(0, r2(dp[:, j], dy[:, j])) for j in range(7)]; e = np.abs(dp - dy); sd = np.nanstd(dy, 0)
        print(f"[ceil] {nm}: psi R2 {r2(P.ravel(), Y.ravel()):.5f} | cons {np.mean(c):.4f} " + " ".join(f"{k} {r2(dp[:, j], dy[:, j]):.3f}" for j, k in enumerate(SC))
              + f" | b0 p5/50/95 {np.percentile(b0s, 5):.2f}/{np.median(b0s):.2f}/{np.percentile(b0s, 95):.2f} | {time.time() - t:.0f}s", flush=True)
        print(f"[ceil] {nm}: median |err| / sd " + " ".join(f"{k} {np.nanmedian(e[:, j]) / sd[j]:.2f}" for j, k in enumerate(SC)), flush=True)


if __name__ == "__main__":
    main()
