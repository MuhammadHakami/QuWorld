"""Doc 35 §42d probe: physics q95 from the PREDICTED psi (TRAIN-only psiF trunks) on VALID + TEST frames, vs the learned q/b heads.
q_phys = |bcoil| * G(psi) * (c0 + c1 betaN_head + c2 li_head) with G = (1/2pi) loop integral dl / (R |grad psi|) at psi_N 0.95
(q_from_psi.G_of); constants fitted on VALID frames, scored on TEST frames; blends with the QH heads' q95 at fixed weights.
    .venv/bin/python paper/strat/fusion/qphys_test.py --n 4000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

import numpy as np
import torch

import fx
import q_from_psi as QP


def _g(p):
    try:
        return QP.G_of(p)
    except Exception:
        return np.nan


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4000); a = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, ens_select as ES
    d = sota.Data(); L = sota.LABELS; rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(np.nonzero(d.split != 0)[0], a.n, replace=False)); shots = [int(s) for s in np.unique(d.shot[rows])]
    mem = [ES.member(str(GB.SOTA / p), GB.fixed(d, "X_magfix"), shots) for p in GB.PSI["psiF"]]
    pos = {}
    for s in shots:
        for k, r in enumerate(np.nonzero(d.shot == s)[0]):
            pos[int(r)] = (s, k)
    pc = np.stack([np.mean([m[pos[int(r)][0]][0][pos[int(r)][1]] for m in mem], 0) for r in rows])
    comps, mean = d.meta["comps"][:ES.KMAX].numpy().astype(np.float64), d.meta["mean"].numpy().astype(np.float64)
    P = (pc @ comps + mean).reshape(-1, 65, 65); T = d.psi_of(rows).reshape(-1, 65, 65)
    with mp.Pool(16) as pool:
        Gp = np.array(pool.map(_g, list(P), chunksize=50)); Gt = np.array(pool.map(_g, list(T), chunksize=50))
    bc = np.abs(np.asarray(GB.fixed(d, "X_ipfix").inputs(rows, np.array([d.fn.index("mag_bcoil")]))[:, 0], np.float64))
    H = GB.heads_gpu(d, rows, "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))); li = GB.heads_gpu(d, rows, "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7)))[:, 6]
    q = d.S[rows, L.index("efit_q95")].astype(np.float64); va = d.split[rows] == 1; te = d.split[rows] == 2
    rep = {}
    for nm, G in (("true_psi", Gt), ("pred_psi", Gp)):
        X = np.stack([bc * G, bc * G * H[:, 1], bc * G * li], 1); ok = np.isfinite(X).all(1) & np.isfinite(q) & (bc > 1)
        fit = ok & va; w = np.linalg.lstsq(X[fit], q[fit], rcond=None)[0]; p = X @ w
        good = ok & (np.abs(p / H[:, 0] - 1) < 0.2)                       # label-free contour-failure guard: > 20% from the head -> head
        rel = np.abs(p / q - 1)                                              # (diagnostic only)
        for part, m in (("VALID", va), ("TEST", te)):
            mm = m & np.isfinite(q); qp = np.where(good, p, H[:, 0])
            r2 = lambda z: 1 - np.mean((q[mm] - z[mm]) ** 2) / np.var(q[mm])
            rep[f"{nm}_{part}"] = {"phys": r2(qp), "head": r2(H[:, 0]), **{f"blend{w_}": r2(w_ * qp + (1 - w_) * H[:, 0]) for w_ in (0.2, 0.35, 0.5)},
                                   "failed": float(1 - good[mm].mean()), "median_rel": float(np.median(rel[mm & good]))}
        print(f"[qphys] {nm}: coef {np.round(w, 5).tolist()} | " + " | ".join(f"{k.split('_')[-1]} " + " ".join(f"{a_} {b_:.5f}" for a_, b_ in v.items()) for k, v in rep.items() if k.startswith(nm)), flush=True)


if __name__ == "__main__":
    main()
