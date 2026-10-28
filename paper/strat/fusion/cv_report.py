"""Fusion §41q proxy CV read-out (user: cross-validation, not seeds): per arm and stratified fold k (STRAT_CV k/5, tag P3<arm>F<k>,
seed 1), the mean of the 7 consistency-scalar R2 on the fold's out-of-fold TRAIN frames (variance pooled over all folds' labels),
and the paired per-fold delta vs a base arm (mean, standard error, t).

    .venv/bin/python paper/strat/fusion/cv_report.py <base> <arm> [<arm> ...]
"""
from __future__ import annotations

import os
import sys

import numpy as np

import fx  # noqa: F401,I001

K = 5
SC = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> None:
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); rt, Yt = lab["rows_train"], lab["y_train"][:, :7].astype(np.float64)
    assert np.all(np.diff(rt) > 0); var = np.nanvar(Yt, 0)
    def score(arm, k, per=False):
        f = fx.OUT / f"head_boundary_v8_{os.environ.get('CV_PFX', 'P3')}{arm}F{k}_s1_oof.npz"
        if not f.exists():
            return None
        z = np.load(f); i = np.searchsorted(rt, z["rows_oof"]); assert np.array_equal(rt[i], z["rows_oof"])
        Y, P = Yt[i], z["pred_oof"][:, :7].astype(np.float64); ok = np.isfinite(Y)
        r = np.array([1 - np.mean((Y[ok[:, j], j] - P[ok[:, j], j]) ** 2) / var[j] for j in range(7)])
        return r if per else float(np.mean(r))
    base, arms = sys.argv[1], sys.argv[2:]
    b = {k: score(base, k) for k in range(K)}
    print(f"[cv] {base:7s} folds {[k for k in b if b[k] is not None]} mean {np.mean([v for v in b.values() if v is not None]):.5f}", flush=True)
    for a in arms:
        d = [score(a, k) - b[k] for k in range(K) if b[k] is not None and score(a, k) is not None]
        if not d:
            continue
        se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"[cv] {a:7s} vs {base}: folds {len(d)} deltas {np.round(d, 5).tolist()} mean {np.mean(d):+.5f} se {se:.5f} t {np.mean(d) / se if se == se and se > 0 else float('nan'):+.1f}", flush=True)
        pd_ = np.mean([score(a, k, True) - score(base, k, True) for k in range(K) if b[k] is not None and score(a, k) is not None], 0)
        print(f"[cv] {a:7s} per scalar: " + " ".join(f"{n} {v:+.5f}" for n, v in zip(SC, pd_)), flush=True)


if __name__ == "__main__":
    main()
