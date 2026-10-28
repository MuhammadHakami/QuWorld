"""Doc 35 §43a: exact X-point labels (v9) - xp_labels.py with every frame's FULL psi (TRAIN rows were labelled from the PCA-256
reconstruction: 1.2 mm median label error). Same saddle rules; psi read per cache shot from its extracted npz.
    .venv/bin/python paper/strat/fusion/xp_labels_exact.py          -> runs/strat/fusion/xp_labels_v9.npz (rows, y = [R_l, Z_l, R_u, Z_u])
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import fx
from d_anatomy import pred_xpoint
from xp_oracle import saddle


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import axis_shift as AX
    from gpu_post import GPUPost
    d = sota.Data(); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); names = d.meta["names"]

    def label(pt):
        o = gp.g.derive(pt); x0 = pred_xpoint(gp, pt, o); xs, r, s = saddle(gp, pt, x0)
        ok = (s & (r < 0.2) & ((xs - x0).norm(dim=1) < 0.03) & torch.isfinite(xs).all(1)).cpu().numpy(); xs = xs.cpu().numpy()
        yy = np.full((len(xs), 4), np.nan, np.float32); lo = ok & (xs[:, 1] < 0); up = ok & (xs[:, 1] >= 0)
        yy[lo, 0:2] = xs[lo]; yy[up, 2:4] = xs[up]; return yy

    y = np.full((len(d.split), 4), np.nan, np.float32); buf_r, buf_p, n = [], [], 0
    starts = np.r_[0, np.nonzero(np.diff(d.shot))[0] + 1]; ends = np.r_[starts[1:], len(d.shot)]

    def flush():
        rr = np.concatenate(buf_r); pp = torch.tensor(np.concatenate(buf_p), device=gp.dev, dtype=torch.float64).reshape(-1, 65, 65)
        y[rr] = label(pp)

    for k, (a, b) in enumerate(zip(starts, ends)):
        j = int(d.shot[a]); psi = np.load(sota.PROC / f"{names[j]}.npz")["psi"].reshape(-1, 4225)
        assert len(psi) == b - a, (names[j], len(psi), b - a)
        buf_r.append(np.arange(a, b)); buf_p.append(psi); n += b - a
        if n >= 2048:
            flush(); buf_r, buf_p, n = [], [], 0
        if k % 500 == 0:
            print(f"[xpl-exact] shot {k}/{len(starts)}", flush=True)
    if buf_r:
        flush()
    old = np.load(fx.OUT / "xp_labels.npz")["y"]; tr = d.split == 0
    for c, (i, nm) in enumerate(((0, "lower"), (2, "upper"))):
        m = tr & np.isfinite(y[:, i]) & np.isfinite(old[:, i]); e = np.hypot(y[m, i] - old[m, i], y[m, i + 1] - old[m, i + 1])
        print(f"[xpl-exact] TRAIN {nm}: |exact - PCA-256 label| median {np.median(e) * 1e3:.2f} mm, p90 {np.quantile(e, 0.9) * 1e3:.2f} mm; "
              f"side agreement {np.mean(np.isfinite(y[tr, i]) == np.isfinite(old[tr, i])):.4f}", flush=True)
    vt = d.split != 0; same = np.allclose(np.nan_to_num(y[vt]), np.nan_to_num(old[vt]), atol=1e-4)
    print(f"[xpl-exact] VALID / TEST rows identical to xp_labels.npz: {same}", flush=True)
    np.savez(fx.OUT / "xp_labels_v9.npz", rows=np.arange(len(d.split)), y=y)
    print("XPL_EXACT_DONE", flush=True)


if __name__ == "__main__":
    main()
