"""Doc 35 §48: data floors for the Lean score ceiling (formal/QuotaScaling/ScoreCeiling.lean).

Question (user, Oct 17): can S reach 0.9999 / 1.0 from the challenge inputs? ScoreCeiling.lean proves that a per-term error floor
becomes a ceiling on S (`r2_ceiling`, `cons_ceiling`, `score_mono`, `unreachable`) and what S >= 1 - eps demands of each term
(`target_each`). This script supplies the floors: the part of each target that is NOT a function of the inputs.

Estimator = the Gamma test (Stefansson, Koncar & Jones 1997; Evans & Jones 2002), cross-shot: for every sampled TRAIN frame, its
k = 1..K nearest neighbours in standardised input space among frames of OTHER shots (a new shot shares no shot-level latent
state with the training shots, so same-shot neighbours would understate the floor); delta_k = mean squared input distance,
gamma_k = mean (y - y_nn)^2 / 2; the intercept of gamma on delta (delta -> 0) estimates E Var(y | x) (the finite version of this
floor is ScoreCeiling.cell_floor / pair_floor). Targets: the 7 consistency scalars, q95, betaN, psi (256 orthonormal PCA
coefficients: |dC|^2 = |P dpsi|^2), and D_LCFS approximated by the discrete symmetric Hausdorff distance between the two
frames' 64-ray LCFS polygons / Rgeo, scaled by 1 / sqrt(2) (two noisy contours vs one). Two stratified shot halves
(head_boundary.cv_shots) give two independent estimates; shot bootstrap for CIs. Inputs = the deployed heads' inputs
(same STRAT_* / QW_MAGFIX environment as the J1RL / V60 heads). Labels only of TRAIN frames.

    env <heads env> QW_GPU_GB=12 runs/capped.sh ceil 40 .venv/bin/python paper/strat/fusion/ceiling.py [every=3] [K=10]
    -> runs/strat/fusion/ceiling.json + CEIL lines
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import fx  # noqa: I001

W = {"psi": 0.55, "qb": 0.15, "d": 0.10, "cons": 0.20}


def knn_cross_shot(X, shot, K, chunk=2048):
    """X (N, F) cuda float32, shot (N,) cuda int -> idx (N, K), d2 (N, K): nearest frames of other shots."""
    n2 = (X * X).sum(1); idx, dd = [], []
    for a in range(0, len(X), chunk):
        q = X[a:a + chunk]; D = n2[a:a + chunk, None] + n2[None] - 2 * q @ X.T
        D[shot[a:a + chunk, None] == shot[None]] = float("inf")
        v, i = torch.topk(D, K, dim=1, largest=False); idx.append(i); dd.append(v.clamp(min=0))
    return torch.cat(idx), torch.cat(dd)


def hausdorff_rays(Ra, Za, ra, Rb, Zb, rb):
    """64-ray LCFS polygons about each axis -> discrete symmetric Hausdorff distance / Rgeo of a (rows with NaN -> NaN)."""
    n = ra.shape[1]; th = 2 * torch.pi * torch.arange(n, device=ra.device) / n; c, s = torch.cos(th), torch.sin(th)
    Pa = torch.stack([Ra[:, None] + ra * c, Za[:, None] + ra * s], -1); Pb = torch.stack([Rb[:, None] + rb * c, Zb[:, None] + rb * s], -1)
    D = torch.cdist(Pa, Pb); h = torch.maximum(D.min(2).values.amax(1), D.min(1).values.amax(1))
    rgeo = 0.5 * (Pa[..., 0].amax(1) + Pa[..., 0].amin(1)); return h / rgeo


def gamma_fit(delta, gam):
    """Least-squares line gam = G + A delta over k; returns (G, A). delta (K,), gam (K, T)."""
    A = np.c_[np.ones_like(delta), delta]; coef, *_ = np.linalg.lstsq(A, gam, rcond=None); return coef[0], coef[1]


def half(d, sota, fm, edge, rows, Y, K, B, rng, log):
    t0 = time.time(); dev = torch.device("cuda")
    X = d.inputs(rows, fm, edge).astype(np.float64); mu, sd = X.mean(0), X.std(0); keep = sd > 1e-9
    Xg = torch.tensor(np.clip((X[:, keep] - mu[keep]) / sd[keep], -10, 10), dtype=torch.float32, device=dev); del X
    sh = torch.tensor(d.shot[rows].astype(np.int64), device=dev)
    idx, d2 = knn_cross_shot(Xg, sh, K); F_ = int(Xg.shape[1]); del Xg
    L = sota.LABELS
    T = {**{f"cons_{nm}": Y[:, j] for j, nm in enumerate(fx.CONS)},
         "q95": d.S[rows, L.index("efit_q95")].astype(np.float64), "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}
    names = list(T); Yt = torch.tensor(np.stack([T[k] for k in names], 1), device=dev)
    C = torch.tensor(np.asarray(d.C[rows]), dtype=torch.float64, device=dev)
    ra = torch.tensor(Y[:, 7:71], dtype=torch.float64, device=dev); Rax, Zax = Yt[:, 0], Yt[:, 1]
    comps = torch.load(sota.CACHE / "meta.pt", weights_only=False); mean, P = comps["mean"].to(dev).double(), comps["comps"].to(dev).double()
    s1 = s2 = 0.0; npx = P.shape[1]
    for a in range(0, len(C), 8192):                                          # scorer normaliser: pooled variance of every pixel
        psi = mean + C[a:a + 8192] @ P; s1 += float(psi.sum()); s2 += float((psi * psi).sum())
    nall = len(C) * npx; psi_var = (s2 - s1 * s1 / nall) / nall
    shots = d.shot[rows]; us, inv = np.unique(shots, return_inverse=True); S_ = len(us)
    gs = np.zeros((S_, K, len(names) + 2)); cnt = np.zeros((S_, K, len(names) + 2)); dl = np.zeros((S_, K))
    for k in range(K):
        j = idx[:, k]; g = 0.5 * (Yt - Yt[j]) ** 2; m = torch.isfinite(g)
        gp = 0.5 * ((C - C[j]) ** 2).sum(1)                                   # psi: |P dpsi|^2 / 2 per frame
        hd = hausdorff_rays(Rax, Zax, ra, Rax[j], Zax[j], ra[j]) / np.sqrt(2)  # D_LCFS proxy (one noisy contour)
        allg = torch.cat([torch.where(m, g, 0.0), gp[:, None], torch.nan_to_num(hd, nan=0.0)[:, None]], 1).cpu().numpy()
        allm = torch.cat([m, torch.ones_like(gp[:, None], dtype=torch.bool), torch.isfinite(hd)[:, None]], 1).cpu().numpy().astype(float)
        for c_ in range(allg.shape[1]):
            gs[:, k, c_] = np.bincount(inv, allg[:, c_], S_); cnt[:, k, c_] = np.bincount(inv, allm[:, c_], S_)
        dl[:, k] = np.bincount(inv, d2[:, k].cpu().numpy().astype(np.float64), S_)
    nfr = np.bincount(inv, minlength=S_).astype(float)
    var = {k: float(np.nanvar(T[k])) for k in names}
    cols = names + ["psi", "dlcfs"]

    def est(wts):
        delta = (wts[:, None] * dl).sum(0) / (wts * nfr).sum()
        gam = (wts[:, None, None] * gs).sum(0) / np.maximum((wts[:, None, None] * cnt).sum(0), 1)
        G, A = gamma_fit(delta, gam); return G, gam[0], delta
    G, g1, delta = est(np.ones(S_)); boots = np.array([est(rng.multinomial(S_, np.ones(S_) / S_).astype(float))[0] for _ in range(B)])
    out = {"frames": len(rows), "shots": S_, "features": F_, "K": K, "delta": delta.tolist(), "psi_var_pix": psi_var, "npix": npx, "t": {}}
    for c_, nm in enumerate(cols):
        if nm == "psi":
            norm = npx * psi_var
        elif nm == "dlcfs":
            norm = 1.0
        else:
            norm = var[nm]
        lo, hi = np.percentile(boots[:, c_], [2.5, 97.5])
        out["t"][nm] = {"floor": float(G[c_]), "lo": float(lo), "hi": float(hi), "gamma1": float(g1[c_]), "norm": float(norm)}
    print(f"[ceil]{log} {len(rows)} frames {S_} shots {F_} inputs, delta_1 {delta[0]:.3f} delta_K {delta[-1]:.3f} (2F = {2 * F_}) "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


def ceiling(o):
    t = o["t"]; r2 = {k: 1 - v["floor"] / v["norm"] for k, v in t.items() if k != "dlcfs"}
    r2_lo = {k: 1 - v["hi"] / v["norm"] for k, v in t.items() if k != "dlcfs"}             # pessimistic floor -> lower ceiling
    cons = float(np.mean([max(0.0, r2[f"cons_{nm}"]) for nm in fx.CONS])); qb = 0.5 * (r2["q95"] + r2["betaN"]); D = max(0.0, t["dlcfs"]["floor"])
    S = W["psi"] * r2["psi"] + W["qb"] * qb + W["d"] * (1 - D) + W["cons"] * cons
    return {"S_max": S, "r2_psi": r2["psi"], "r2_qb": qb, "dlcfs": D, "cons": cons, "r2": r2, "r2_lo": r2_lo,
            "deficit": {"psi": W["psi"] * (1 - r2["psi"]), "qb": W["qb"] * (1 - qb), "d": W["d"] * D, "cons": W["cons"] * (1 - cons)}}


def main() -> None:
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 3; K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if "QW_GPU_GB" in os.environ:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["QW_GPU_GB"]) / 120.0))
    torch.backends.cuda.matmul.allow_tf32 = False
    import head_boundary as HB
    import hpo_qpikan as H
    d, sota, c, fm, edge, lab = H.data(); tr, Y = lab["rows_train"], lab["y_train"].astype(np.float64)
    rng = np.random.default_rng(0); res = {}
    for h in (0, 1):
        shots = HB.cv_shots(d, tr, Y, h, 2); m = np.isin(d.shot[tr], shots)
        o = half(d, sota, fm, edge, tr[m][::every], Y[m][::every], K, 200, rng, f" half {h}"); o["ceiling"] = ceiling(o); res[f"half{h}"] = o
        cl = o["ceiling"]
        print(f"CEIL half {h}: S_max {cl['S_max']:.5f} | psi {cl['r2_psi']:.6f} qb {cl['r2_qb']:.5f} D {cl['dlcfs']:.5f} cons {cl['cons']:.5f} | "
              f"deficit psi {cl['deficit']['psi']:.5f} qb {cl['deficit']['qb']:.5f} D {cl['deficit']['d']:.5f} cons {cl['deficit']['cons']:.5f}", flush=True)
        print("   per target R2 ceiling [pessimistic]: " + " ".join(f"{k} {v:.5f} [{cl['r2_lo'][k]:.5f}]" for k, v in cl["r2"].items()), flush=True)
    (fx.OUT / "ceiling.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
