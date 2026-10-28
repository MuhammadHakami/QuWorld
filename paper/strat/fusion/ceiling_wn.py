"""Doc 35 §48b: white-noise floors for the Lean score ceiling (formal/QuotaScaling/ScoreCeiling.lean), TRAIN labels only.

§48a (ceiling.py, cross-shot Gamma test) is rejected: with 571 inputs the nearest other-shot frame is at a mean squared standardised
distance of 30 (random pairs 1142), the delta -> 0 extrapolation fails (psi ceiling 0.9962 < the 0.99986 we already reach; volume /
q95 intercepts negative). Here the floor is the part of each label that changes independently from one EFIT time slice to the next
(EFIT fits every slice separately from magnetic probes that are not challenge inputs): on equally spaced triples inside a shot,
d = y(t+1) - 2 y(t) + y(t-1) carries 6 sigma^2 of independent slice noise plus the signal's curvature. Two estimates:
  robust: sigma = 1.4826 median|d| / sqrt(6)  (MAD; sparse ELM / sawtooth jumps do not move it; curvature only inflates it)
  two-lag: sigma^2 = (16 V1 - V2) / 90, V_h = mean d_h^2 on spacing h and 2h (removes an h^4 curvature term; jumps inflate it)
Floor = sigma^2 per target; psi = sum over the 256 orthonormal PCA coefficients (scorer normaliser: pooled pixel variance);
D_LCFS = mean Hausdorff(contour(t), mid(t-1, t+1)) / Rgeo / sqrt(1.5) on the 64-ray polygons (approximate).
These floors bound any inputs-only predictor IF the slice noise is independent of the inputs (it comes from probes the inputs do not
contain); they ignore information missing at the shot level, so the ceiling they give is an UPPER bound on what is reachable.

    .venv/bin/python paper/strat/fusion/ceiling_wn.py   -> runs/strat/fusion/ceiling_wn.json + WN lines
"""
from __future__ import annotations

import json

import numpy as np

import fx  # noqa: I001

W = {"psi": 0.55, "qb": 0.15, "d": 0.10, "cons": 0.20}


def triples(shot, t, lag):
    """Indices (a, b, c) of frames t-lag, t, t+lag inside one shot with equal spacing (1 %)."""
    i = np.arange(lag, len(t) - lag); a, c = i - lag, i + lag
    ok = (shot[a] == shot[i]) & (shot[c] == shot[i]); h1, h2 = t[i] - t[a], t[c] - t[i]
    ok &= (h1 > 0) & (np.abs(h1 - h2) <= 0.01 * h1); return a[ok], i[ok], c[ok]


def hausdorff(Ra, Za, ra, Rb, Zb, rb):
    n = ra.shape[1]; th = 2 * np.pi * np.arange(n) / n; c, s = np.cos(th), np.sin(th)
    Pa = np.stack([Ra[:, None] + ra * c, Za[:, None] + ra * s], -1); Pb = np.stack([Rb[:, None] + rb * c, Zb[:, None] + rb * s], -1)
    D = np.sqrt(((Pa[:, :, None] - Pb[:, None]) ** 2).sum(-1)); h = np.maximum(D.min(2).max(1), D.min(1).max(1))
    return h / (0.5 * (Pa[..., 0].max(1) + Pa[..., 0].min(1)))


def main() -> None:
    import sota
    d = sota.Data(); lab = np.load(fx.OUT / "boundary_labels_v8.npz"); rows, Y = lab["rows_train"], lab["y_train"].astype(np.float64)
    o = np.argsort(rows, kind="stable"); rows, Y = rows[o], Y[o]
    shot = d.shot[rows]; t = np.asarray(d.X[rows, d.fn.index("t")], np.float64); L = sota.LABELS
    T = {**{f"cons_{nm}": Y[:, j] for j, nm in enumerate(fx.CONS)},
         "q95": d.S[rows, L.index("efit_q95")].astype(np.float64), "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}
    C = np.asarray(d.C[rows], np.float64)
    a1, b1, c1 = triples(shot, t, 1); a2, b2, c2 = triples(shot, t, 2)
    print(f"[wn] {len(rows)} TRAIN frames, {len(np.unique(shot))} shots; equally spaced triples lag 1 {len(b1)}, lag 2 {len(b2)}; "
          f"median spacing {np.median(t[b1] - t[a1]):.1f} ms", flush=True)

    def floors(y):
        d1 = y[c1] - 2 * y[b1] + y[a1]; d2 = y[c2] - 2 * y[b2] + y[a2]; d1, d2 = d1[np.isfinite(d1)], d2[np.isfinite(d2)]
        rob = (1.4826 * np.median(np.abs(d1))) ** 2 / 6; two = max(0.0, (16 * np.mean(d1 ** 2) - np.mean(d2 ** 2)) / 90)
        return rob, two
    res = {"frames": len(rows), "t": {}}
    for k, y in T.items():
        rob, two = floors(y); v = float(np.nanvar(y)); res["t"][k] = {"var": v, "robust": rob, "twolag": two}
    fr = np.array([floors(C[:, j]) for j in range(C.shape[1])])
    meta = d.meta; mean, P = meta["mean"].double().numpy(), meta["comps"].double().numpy(); npx = P.shape[1]
    sub = np.random.default_rng(0).choice(len(C), min(60000, len(C)), replace=False); psi = mean + C[sub] @ P
    psi_var = float(psi.var()); res["t"]["psi"] = {"var": npx * psi_var, "robust": float(fr[:, 0].sum()), "twolag": float(fr[:, 1].sum())}
    ra = Y[:, 7:71]; Rx, Zx = Y[:, 0], Y[:, 1]; s = np.random.default_rng(1).choice(len(b1), min(200000, len(b1)), replace=False)
    a_, b_, c_ = a1[s], b1[s], c1[s]
    h = np.concatenate([hausdorff(Rx[b_[i:i + 8192]], Zx[b_[i:i + 8192]], ra[b_[i:i + 8192]], 0.5 * (Rx[a_[i:i + 8192]] + Rx[c_[i:i + 8192]]),
                                  0.5 * (Zx[a_[i:i + 8192]] + Zx[c_[i:i + 8192]]), 0.5 * (ra[a_[i:i + 8192]] + ra[c_[i:i + 8192]]))
                        for i in range(0, len(b_), 8192)]) / np.sqrt(1.5)                     # chunks: 8192 x 64 x 64 pairs
    h = h[np.isfinite(h)]; res["t"]["dlcfs"] = {"mean": float(h.mean()), "median": float(np.median(h))}
    for est in ("robust", "twolag"):
        r2 = {k: 1 - v[est] / v["var"] for k, v in res["t"].items() if k != "dlcfs"}
        cons = float(np.mean([max(0.0, r2[f"cons_{nm}"]) for nm in fx.CONS])); qb = 0.5 * (r2["q95"] + r2["betaN"])
        D = res["t"]["dlcfs"]["median" if est == "robust" else "mean"]
        S = W["psi"] * r2["psi"] + W["qb"] * qb + W["d"] * (1 - D) + W["cons"] * cons
        res[est] = {"S_max": S, "r2": r2, "cons": cons, "qb": qb, "dlcfs": D,
                    "deficit": {"psi": W["psi"] * (1 - r2["psi"]), "qb": W["qb"] * (1 - qb), "d": W["d"] * D, "cons": W["cons"] * (1 - cons)}}
        df = res[est]["deficit"]
        print(f"WN {est}: S_max {S:.6f} | psi {r2['psi']:.7f} qb {qb:.6f} D {D:.5f} cons {cons:.6f} | deficit psi {df['psi']:.6f} "
              f"qb {df['qb']:.6f} D {df['d']:.6f} cons {df['cons']:.6f}", flush=True)
        print("   R2 ceilings: " + " ".join(f"{k} {v:.6f}" for k, v in r2.items()), flush=True)
    (fx.OUT / "ceiling_wn.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
