"""Per consistency scalar on the 180-shot VALID sample: R^2 (fixed normaliser, as the scorer) of
(a) the 6-seed v8 shape heads, (b) the entry trunk's EFIT axis heads, (c) the frozen pipeline's achieved values,
(d) the raw entry. Tells whether the remaining consistency loss is head accuracy or warp shortfall.

    .venv/bin/python paper/strat/fusion/weak_heads.py
"""
from __future__ import annotations

import pickle

import numpy as np

import fx  # noqa: I001
import strategies as ST
from gpu_pipe import NAMES


def main() -> None:
    sota, d = fx.data(); shots, grp = ST.sample(); pe, nm = ST.load_entry(); f, comps, mean, H = ST.entry_preds(d, sota)
    fz = pickle.load(open(fx.OUT / "parts_BW-Q-bkl3_v8_6s.pkl", "rb"))
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    rows = np.concatenate([fx.valid_rows(d, s) for s in shots]); ph = Ph[[pos[int(r)] for r in rows]]
    gt = np.concatenate([pe[s]["cgt"] for s in shots]); var = np.asarray(nm["c_var"])
    sc = np.concatenate([f(s)[1].mean(0) for s in shots]); tr = np.c_[sc[:, H.index("efit_r_axis")], sc[:, H.index("efit_z_axis")]]

    def r2(p, k):
        m = np.isfinite(gt[:, k]) & np.isfinite(p); return 1 - np.mean((gt[m, k] - p[m]) ** 2) / var[k]
    print(f"{'scalar':8s} {'v8 head':>9s} {'trunk':>9s} {'frozen':>9s} {'entry':>9s}")
    for k, n in enumerate(NAMES):
        fr = np.concatenate([fz[s]["cpred"][:, k] for s in shots]); en = np.concatenate([pe[s]["cpred"][:, k] for s in shots])
        t = f"{r2(tr[:, k], k):9.4f}" if k < 2 else f"{'':9s}"
        print(f"{n:8s} {r2(ph[:, k], k):9.4f} {t} {r2(fr, k):9.4f} {r2(en, k):9.4f}")


if __name__ == "__main__":
    main()
