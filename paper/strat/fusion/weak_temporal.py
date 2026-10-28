"""Are per-frame prediction errors white in time? Centered moving-average smoothing within each shot (window w frames,
edges shrink) of (a) the entry's PCA coefficients, (b) q95 / betaN, (c) the v8 shape heads; R^2 against the targets on
the 180-shot VALID sample. Error autocorrelation at lag 1 is printed as well.

    .venv/bin/python paper/strat/fusion/weak_temporal.py
"""
from __future__ import annotations

import numpy as np

import fx  # noqa: I001
import strategies as ST
from gpu_pipe import NAMES


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w); h = w // 2
    num = np.apply_along_axis(lambda v: np.convolve(np.pad(v, h, mode="constant"), k, "valid"), 0, x)
    den = np.convolve(np.pad(np.ones(len(x)), h, mode="constant"), k, "valid")[:, None]
    return num / den


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    L = sota.LABELS
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    var_c = np.asarray(nm["c_var"])
    for w in (1, 3, 5, 7):
        ec, eq, eb, ehs, ac = 0.0, 0.0, 0.0, np.zeros(7), []
        cvar = None; n = 0; nh = np.zeros(7)
        for s in shots:
            rows = fx.valid_rows(d, s); cs, ss = f(s); c = smooth(cs.mean(0), w); sc = smooth(ss.mean(0), w)
            ct = np.asarray(d.C[rows][:, :c.shape[1]], np.float64); e = c - ct; ec += (e[:, :32] ** 2).sum(0); n += len(rows)
            if w == 1 and len(e) > 2:
                ac.append(np.corrcoef(e[:-1, 0], e[1:, 0])[0, 1])
            eq += ((sc[:, H.index("efit_q95")] - d.S[rows, L.index("efit_q95")]) ** 2).sum()
            eb += ((sc[:, H.index("efit_beta_n")] - d.S[rows, L.index("efit_beta_n")]) ** 2).sum()
            ph = smooth(Ph[[pos[int(r)] for r in rows]][:, :7], w); gt = pe[s]["cgt"]; m = np.isfinite(gt)
            ehs += np.where(m, (ph - np.where(m, gt, 0)) ** 2, 0).sum(0); nh += m.sum(0)
        r2h = 1 - ehs / nh / var_c
        print(f"[temporal] w={w}: coef sq-err [0,32) {ec.sum() / n:.5f} (c0 {ec[0] / n:.5f}); q95 mse {eq / n:.5f} betaN mse {eb / n:.6f}; "
              f"heads R2 " + " ".join(f"{k}:{v:.4f}" for k, v in zip(NAMES, r2h)), flush=True)
        if w == 1:
            print(f"[temporal] c0 error lag-1 autocorrelation median over shots {np.nanmedian(ac):.3f}", flush=True)


if __name__ == "__main__":
    main()
