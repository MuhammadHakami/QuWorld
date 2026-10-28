"""Doc 35 §46: validate the free-boundary solver on labelled DIII-D frames (equilibria from another machine; no MAST data).
psi R2 (pooled, global sign normalised like the scorer), the 7 consistency scalars of the solver's psi vs those of EFIT psi (GPU
scorer twin, DIII-D envelope), q95 / betaN not yet. Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python paper/strat/fusion/fb/test_d3d.py [n_shots] [ecoil_turns] [b0] [am] [an]
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
from solve import Solver  # noqa: E402

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def r2(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    return 1 - np.sum((p[ok] - y[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20; et = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    b0, am, an = (float(x) for x in (sys.argv[3:6] if len(sys.argv) > 5 else (0.5, 1.0, 2.0)))
    t = time.time(); m = geom.build("d3d", ecoil_turns=et); print(f"[fb] DIII-D geometry {time.time() - t:.0f}s circuits {m.circuits}", flush=True)
    fs = data.files("diii_d_train"); rng = np.random.default_rng(0); pick = sorted(rng.choice(len(fs), n, replace=False))
    D = data.load("diii_d_train", m.circuits, "d3d", [fs[i] for i in pick], labels=True)
    ok = np.isfinite(D["I"]).all(1) & np.isfinite(D["Ip"])
    print(f"[fb] frames {len(D['Ip'])} finite {ok.sum()}", flush=True)
    sv = Solver(m); t = time.time(); outs = []
    I = torch.tensor(np.nan_to_num(D["I"]), dtype=torch.float64); Ip = torch.tensor(np.nan_to_num(D["Ip"]))
    for a in range(0, len(Ip), 512):
        o = sv.solve(I[a:a + 512], Ip[a:a + 512], b0=b0, am=am, an=an, R_init=1.7)
        outs.append({k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in o.items() if k in ("psi", "ok", "dJ", "n_iter", "hist")})
        print(f"[fb] batch {a}: iters {o['n_iter']} dJ med {float(o['dJ'].median()):.2e} max {float(o['dJ'].max()):.2e} ok {float(o['ok'].float().mean()):.3f} | {time.time() - t:.0f}s", flush=True)
    # DIII-D EFIT stores psi with the axis at the minimum: psi_EFIT = -sign(Ip) * physical flux (checked on 1,548 frames)
    s_ip = -np.sign(D["Ip"])[:, None, None]
    P = s_ip * np.concatenate([o["psi"] for o in outs]).astype(np.float64); Y = D["psi"].astype(np.float64)
    pc = s_ip * sv.coil_psi(I).cpu().numpy()
    print(f"[fb] psi R2 pooled {r2(P.ravel(), Y.ravel()):.5f}; coil-only R2 {r2(pc.ravel(), Y.ravel()):.5f}", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
    import axis_shift as AX
    from gpu_scorer import GPUScorer
    gR, gZ, mask = AX.grid(); sc = GPUScorer(gR, gZ, mask); dp, dy = [], []
    for a in range(0, len(P), 1024):
        op = sc.derive(torch.tensor(P[a:a + 1024])); oy = sc.derive(torch.tensor(Y[a:a + 1024]))
        dp.append(np.stack([op[k].cpu().numpy() for k in SC], 1)); dy.append(np.stack([oy[k].cpu().numpy() for k in SC], 1))
    dp, dy = np.concatenate(dp), np.concatenate(dy)
    print("[fb] consistency R2: " + " ".join(f"{k} {r2(dp[:, j], dy[:, j]):.4f}" for j, k in enumerate(SC)), flush=True)
    print("[fb] mean |err|: " + " ".join(f"{k} {np.nanmean(np.abs(dp[:, j] - dy[:, j])):.4f}" for j, k in enumerate(SC)), flush=True)
    np.savez(os.path.join(HERE, "..", "..", "..", "..", "runs", "strat", "fusion", "fb_d3d_test.npz"), P=P.astype(np.float32), Y=Y.astype(np.float32), dp=dp, dy=dy)


if __name__ == "__main__":
    main()
