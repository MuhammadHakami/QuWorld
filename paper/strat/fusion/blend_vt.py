"""Doc 35 §42m diagnostic: are the shape heads and a raw psi's derived scalars complementary? On all VALID + TEST frames, per consistency
scalar: R2 of the head ensemble, of each parts arm's derived value (cpred, official references cgt), and of the blend w * head +
(1 - w) * derived with w fitted on VALID shots and read out on TEST shots (never used for selection). If a blend beats the heads,
the post-processing chain should target the blend instead of the heads.
    .venv/bin/python paper/strat/fusion/blend_vt.py <'+'-joined heads> <parts tag> [...]
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import torch

import fx

SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
WS = np.round(np.linspace(0, 1, 21), 2)


def r2(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    return 1 - np.mean((p[ok] - y[ok]) ** 2) / np.var(y[ok])


def main() -> None:
    heads, tags = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); shots = [int(s) for s in np.unique(d.shot[d.split != 0])]
    allr = np.concatenate([fx.scored_rows(d, s) for s in shots]); sp = d.split[allr]
    H = GB.heads_gpu(d, allr, heads)[:, :7]
    for tag in tags:
        P = pickle.load(open(fx.OUT / f"parts_VT-{tag}.pkl", "rb"))
        C = np.concatenate([P[s]["cpred"] for s in shots]); Y = np.concatenate([P[s]["cgt"] for s in shots])
        mh, md, mb = [], [], []
        for j, n in enumerate(SC):
            va, te = sp == 1, sp == 2
            fit = [r2(w * H[va, j] + (1 - w) * C[va, j], Y[va, j]) for w in WS]; w = WS[int(np.argmax(fit))]
            rh, rd, rb = r2(H[te, j], Y[te, j]), r2(C[te, j], Y[te, j]), r2(w * H[te, j] + (1 - w) * C[te, j], Y[te, j])
            ok = np.isfinite(H[:, j]) & np.isfinite(C[:, j]) & np.isfinite(Y[:, j])
            rho = np.corrcoef(H[ok, j] - Y[ok, j], C[ok, j] - Y[ok, j])[0, 1]
            mh.append(rh); md.append(rd); mb.append(rb)
            print(f"[blend] {tag} {n:8s} TEST head {rh:.5f} derived {rd:.5f} blend(w {w:.2f}) {rb:.5f} gain {rb - rh:+.5f} | err corr {rho:.3f}",
                  flush=True)
        print(f"[blend] {tag} mean-7 TEST head {np.mean(mh):.5f} derived {np.mean(md):.5f} blend {np.mean(mb):.5f} gain {np.mean(mb) - np.mean(mh):+.5f}",
              flush=True)


if __name__ == "__main__":
    main()
