"""Doc 35 §46 algorithm check on labelled DIII-D frames: solver with radial control to the EFIT outer-midplane LCFS radius (the
quantity MAST's midplane Thomson can supply), driven by (a) EFIT's own external flux or (b) the measured coils only. Psi R2 and the 7
consistency scalars vs EFIT. Validation only (DIII-D labels).
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python paper/strat/fusion/fb/test_rout.py [n_shots] [b0 am an] [n_iter] [w]
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


def efit_rout(sv, Y, ipi):
    phi = -torch.sign(torch.tensor(ipi, device="cuda", dtype=torch.float32))[:, None, None] * torch.tensor(Y, device="cuda", dtype=torch.float32)
    phi = -phi * 0 + torch.tensor(np.sign(ipi), device="cuda", dtype=torch.float32)[:, None, None] * torch.tensor(Y, device="cuda", dtype=torch.float32)
    iz, ir, _ = sv.axis(phi, torch.full((len(Y),), 1.7, device="cuda"), torch.zeros(len(Y), device="cuda"))
    h = sv.reach(phi, iz, ir)
    bnd = torch.where(sv.out[None], h, torch.full_like(h, -np.inf)).flatten(1).max(1).values
    P = (h > bnd[:, None, None]) & sv.lim[None]
    return sv.r_out(phi, P, iz, bnd).cpu().numpy()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    b0, am, an = (float(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (0.2, 2.0, 1.0)))
    nit = int(sys.argv[5]) if len(sys.argv) > 5 else 120; w = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    m = geom.build("d3d"); fs = data.files("diii_d_train"); rng = np.random.default_rng(0); pick = sorted(rng.choice(len(fs), n, replace=False))
    D = data.load("diii_d_train", m.circuits, "d3d", [fs[i] for i in pick], labels=True)
    Y = D["psi"].astype(np.float64); J, ext, ipi = efit_split(Y, m); sv = Solver(m)
    ro = np.concatenate([efit_rout(sv, Y[a:a + 512], ipi[a:a + 512]) for a in range(0, len(Y), 512)])
    print(f"[rout] frames {len(Y)}; EFIT R_out {np.nanmin(ro):.3f}-{np.nanmax(ro):.3f}", flush=True)
    # coils-only: the coil flux has the physical sign; EFIT psi = -sign(Ip) * physical -> flip the coil flux into the EFIT convention
    s_ip = -np.sign(D["Ip"]); s_ip[s_ip == 0] = 1
    pcoil = s_ip[:, None, None] * sv.coil_psi(torch.tensor(np.nan_to_num(D["I"]))).cpu().numpy()
    import axis_shift as AX
    from gpu_scorer import GPUScorer
    gR, gZ, mask = AX.grid(); sc = GPUScorer(gR, gZ, mask)
    for nm, pc in (("efit-ext", ext), ("coils", pcoil)):
        t = time.time(); P = []; bv = []
        for a in range(0, len(Y), 512):
            o = sv.solve(None, ipi[a:a + 512] / 1e3, b0=b0, am=am, an=an, R_init=1.7, pc=pc[a:a + 512], R_out=ro[a:a + 512], n_iter=nit, w=w, kp=0.15, ki=0.01)
            P.append(o["psi"].cpu().numpy()); bv.append(o["Bv"].cpu().numpy())
        P = np.concatenate(P).astype(np.float64); bv = np.concatenate(bv)
        dp, dy = [], []
        for a in range(0, len(P), 1024):
            op = sc.derive(torch.tensor(P[a:a + 1024])); oy = sc.derive(torch.tensor(Y[a:a + 1024]))
            dp.append(np.stack([op[k].cpu().numpy() for k in SC], 1)); dy.append(np.stack([oy[k].cpu().numpy() for k in SC], 1))
        dp, dy = np.concatenate(dp), np.concatenate(dy); c = [max(0, r2(dp[:, j], dy[:, j])) for j in range(7)]
        print(f"[rout] {nm}: psi R2 {r2(P.ravel(), Y.ravel()):.5f} | cons {np.mean(c):.4f} " + " ".join(f"{k} {r2(dp[:, j], dy[:, j]):.3f}" for j, k in enumerate(SC))
              + f" | Bv med {np.median(bv):+.4f} T | {time.time() - t:.0f}s", flush=True)
        e = np.abs(dp - dy); sd = np.nanstd(dy, 0)
        print(f"[rout] {nm}: median |err| / sd " + " ".join(f"{k} {np.nanmedian(e[:, j]) / sd[j]:.2f}" for j, k in enumerate(SC))
              + f" | frames |kappa err| > 0.2: {np.mean(e[:, 2] > 0.2):.3f}; R2 without the worst 5%: "
              + " ".join(f"{k} {r2(np.where(e[:, j] <= np.nanpercentile(e[:, j], 95), dp[:, j], np.nan), dy[:, j]):.2f}" for j, k in enumerate(SC)), flush=True)


if __name__ == "__main__":
    main()
