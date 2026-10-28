"""Exploratory per-target blend of shape-head families on VT, cross-fitted by shot (sol's guard: VT is not disjoint from the
weight fit here, so this is labelled exploratory; a deployable blend would be fitted on TRAIN shot-OOF predictions).

Every family = mean of its TRAIN-only seeds (gpu_build.heads_gpu). For each of the 7 consistency scalars, simplex weights over the
families (register family >= REG_MIN: the hybrid share is kept) are fitted by least squares on half of the VT shots and read on
the other half (and swapped), for R repeated random shot halvings. Read-out: mean R2 of the cross-fitted blend vs the deployed
style A = J1RL_s1*3 + J1RL_s2*3 + J1QPIpi_s1*1 on the same shots (paired by repetition), and the averaged weights.

    .venv/bin/python paper/strat/fusion/blend_cv.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from scipy.optimize import minimize

import fx
from vt_eval import SC, r2, shot_sums

H = "head_boundary_v8_"
FAM = {"J1RL": ["J1RL_s1", "J1RL_s2"], "J1QPIpi": ["J1QPIpi_s1", "J1QPIpi_s2", "J1QPIpi_s3"], "J1": [f"J1_s{s}" for s in range(1, 7)],
       "J1QRL": ["J1QRL_s1"], "J1RLM": ["J1RLM_s1"], "J1RLX": ["J1RLX_s1"], "J1THP": ["J1THP_s1"], "J2": ["J2_s1"]}
REG = ("J1QPIpi", "J1QRL")                                                  # register families (30-atom register heads)
REG_MIN, REPS = 1 / 7, 20
BASE = f"{H}J1RL_s1*3+{H}J1RL_s2*3+{H}J1QPIpi_s1*1"


def fit_w(P, y):
    """Simplex least squares for one target: P (n, F), y (n,); register families share >= REG_MIN."""
    F = P.shape[1]; ireg = [i for i, f in enumerate(FAM) if f in REG]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, {"type": "ineq", "fun": lambda w: w[ireg].sum() - REG_MIN}]
    v = y.var() + 1e-30; obj = lambda w: ((P @ w - y) ** 2).mean() / v  # noqa: E731   (scaled: 1 - R2-like, SLSQP tolerances)
    jac = lambda w: 2.0 * P.T @ (P @ w - y) / len(y) / v  # noqa: E731
    r = minimize(obj, np.full(F, 1.0 / F), jac=jac, bounds=[(0.0, 1.0)] * F, constraints=cons, method="SLSQP",
                 options={"maxiter": 500, "ftol": 1e-12})
    return r.x


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    d = sota.Data(); vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); rows, y = vt["rows"], vt["y"].astype(np.float64)[:, :7]
    sh = d.shot[rows]; us, inv = np.unique(sh, return_inverse=True)
    P = np.stack([GB.heads_gpu(d, rows, "+".join(H + m for m in mem))[:, :7] for mem in FAM.values()], 2)    # (n, 7, F)
    base = GB.heads_gpu(d, rows, BASE)[:, :7]; ok = np.isfinite(y).all(1) & np.isfinite(P).all((1, 2)) & np.isfinite(base).all(1)
    rng = np.random.default_rng(0); dl, du, Ws = [], [], []
    for rep in range(REPS):
        half = rng.permutation(len(us)) < len(us) // 2; fold = half[inv]; pb = np.full_like(base, np.nan)
        for f in (True, False):
            tr, te = ok & (fold == f), fold != f
            W = np.stack([fit_w(P[tr, j], y[tr, j]) for j in range(7)]); Ws.append(W)
            pb[te] = np.einsum("njf,jf->nj", P[te], W)
        bb = np.where(np.isfinite(pb), base, np.nan)                                 # same frames for both arms
        uni = np.where(np.isfinite(pb), P.mean(2), np.nan)                           # equal-weight family average (no fit)
        d1 = r2(shot_sums(sh, y, pb)).mean() - r2(shot_sums(sh, y, bb)).mean(); dl.append(d1)
        du.append(r2(shot_sums(sh, y, uni)).mean() - r2(shot_sums(sh, y, bb)).mean())
        print(f"[blend] rep {rep}: cross-fitted blend - A = {d1:+.6f} | uniform families - A = {du[-1]:+.6f}", flush=True)
    W = np.mean(Ws, 0); out = {"delta_mean": float(np.mean(dl)), "delta_sd": float(np.std(dl)), "reps": REPS,
                               "weights": {t: dict(zip(FAM, np.round(W[j], 3).tolist())) for j, t in enumerate(SC)}}
    out["uniform_delta_mean"] = float(np.mean(du))
    print(f"[blend] cross-fitted per-target blend vs A: {np.mean(dl):+.6f} (sd over halvings {np.std(dl):.6f}); "
          f"uniform family average vs A: {np.mean(du):+.6f}", flush=True)
    for j, t in enumerate(SC):
        print(f"[blend] {t}: " + " ".join(f"{f} {W[j, i]:.2f}" for i, f in enumerate(FAM)), flush=True)
    (fx.OUT / "blend_cv.json").write_text(json.dumps(out, indent=1)); print("BLEND_CV_DONE", flush=True)


if __name__ == "__main__":
    main()
