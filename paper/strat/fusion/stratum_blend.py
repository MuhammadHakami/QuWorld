"""Doc 35 §48e (Lean NoiseRobust.stratum_blend_le: per-stratum blend weights are never worse than one pooled weight on the fitted set).
Head-level read on the 180-shot VALID sample (TRAIN-only twins, the F18K shape-head composition): groups MLP (J1RL / J1RLCS on reversed Ip,
24), MLP + register (J1QPIpi s1-2 x2, J1QHCS s1-3 x4), VisionHOPE + register (V60 s1-2 x6.75). Strata = Ip sign x shot phase (first 10% /
middle / last 10% of the shot's frames). Per stratum, nonnegative group weights summing to 1 minimise the standardised 7-scalar squared
error (+ rays at their loss weight); honest estimate by 5-fold shot-level cross-fitting (weights fitted on 4/5 of the shots, applied to the
rest). Read-out: mean-7 R2 and rays mean R2, fixed F18K weights vs pooled-refit vs per-stratum-refit.
    .venv/bin/python paper/strat/fusion/stratum_blend.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

import fx  # noqa: I001

H = fx.OUT


def pv(stem):
    return np.load(H / f"head_boundary_v8_{stem}.npz")["pred_valid"].astype(np.float64)


def main() -> None:
    import sota
    lab = np.load(H / "boundary_labels_v8.npz"); rows, Y = lab["rows_valid"], lab["y_valid"].astype(np.float64)
    d = sota.Data(); shot = d.shot[rows]; t = np.asarray(d.X[rows, d.fn.index("t")], np.float64)
    ip = np.asarray(d.X[rows, d.fn.index("mag_plasma_current")], np.float64)
    us, inv = np.unique(shot, return_inverse=True)
    sgn = np.sign(np.bincount(inv, ip) / np.bincount(inv))[inv]
    t0 = np.minimum.reduceat(t, np.r_[0, np.nonzero(np.diff(shot))[0] + 1]) if (np.diff(shot) >= 0).all() else None
    tmin = np.array([t[inv == k].min() for k in range(len(us))])[inv]; tmax = np.array([t[inv == k].max() for k in range(len(us))])[inv]
    ph = (t - tmin) / np.maximum(tmax - tmin, 1e-9); phase = np.where(ph < 0.1, 0, np.where(ph > 0.9, 2, 1))
    stratum = (sgn < 0).astype(int) * 3 + phase
    mlp = np.where((sgn < 0)[:, None], pv("J1RLCS_s1"), pv("J1RL_s1"))
    reg = (2 * pv("J1QPIpi_s1") + 2 * pv("J1QPIpi_s2") + 4 * (pv("J1QHCS_s1") + pv("J1QHCS_s2") + pv("J1QHCS_s3"))) / 16
    vh = 0.5 * (pv("V60_s1") + pv("V60_s2"))
    G = np.stack([mlp, reg, vh], 0); w0 = np.array([24, 16, 13.5]) / 53.5
    sd = np.nanstd(Y, 0); lw = np.ones(71); lw[:7] = 9.142857
    ok = np.isfinite(Y)

    def r2(P, m):
        out = []
        for k in range(71):
            mk = m & ok[:, k]; out.append(1 - np.mean((Y[mk, k] - P[mk, k]) ** 2) / np.var(Y[mk, k]))
        return np.array(out)

    def fit_w(m):
        E = [(np.where(ok[m], G[g][m] - 0, 0)) for g in range(3)]
        A = np.stack([((G[g][m] - 0) / sd * np.sqrt(lw)).ravel() for g in range(3)], 1); b = (np.where(ok[m], Y[m], 0) / sd * np.sqrt(lw)).ravel()
        msk = ok[m].ravel(); A, b = A[msk], b[msk]
        A2 = np.vstack([A, 1e3 * np.ones((1, 3))]); b2 = np.r_[b, 1e3]          # soft sum-to-one, nonnegative
        w, _ = nnls(A2, b2); return w / w.sum()
    rng = np.random.default_rng(0); fold = rng.permutation(len(us)) % 5; f_row = fold[inv]
    P_fix = np.tensordot(w0, G, 1); P_pool = np.zeros_like(Y); P_str = np.zeros_like(Y); W_str = {}
    for f in range(5):
        tr, te = f_row != f, f_row == f
        wp = fit_w(tr); P_pool[te] = np.tensordot(wp, G[:, te], 1)
        for s in range(6):
            mtr, mte = tr & (stratum == s), te & (stratum == s)
            if mte.any():
                ws = fit_w(mtr) if mtr.sum() > 200 else wp; W_str.setdefault(s, []).append(ws); P_str[mte] = np.tensordot(ws, G[:, mte], 1)
    allm = np.ones(len(Y), bool)
    for nm, P in (("fixed F18K", P_fix), ("pooled refit (CV)", P_pool), ("per-stratum refit (CV)", P_str)):
        r = r2(P, allm); print(f"STRAT {nm:24s} mean-7 {r[:7].mean():.5f} rays {r[7:].mean():.5f} | " + " ".join(f"{n} {v:.5f}" for n, v in zip(fx.CONS, r[:7])), flush=True)
    names = ["+Ip ramp-up", "+Ip flat", "+Ip ramp-down", "-Ip ramp-up", "-Ip flat", "-Ip ramp-down"]
    for s in range(6):
        m = stratum == s
        if m.any():
            print(f"   stratum {names[s]:14s} frames {m.sum():6d} | mean weights (MLP, MLP+reg, VH+reg) {np.round(np.mean(W_str.get(s, [w0]), 0), 3).tolist()} | "
                  f"mean-7 fixed {r2(P_fix, m)[:7].mean():.5f} strat {r2(P_str, m)[:7].mean():.5f}", flush=True)


if __name__ == "__main__":
    main()
