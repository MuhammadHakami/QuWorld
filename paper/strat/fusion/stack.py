"""§37 fusion, strategy STACK: per-output non-negative blend weights over the 7 entry members, fit on the TRAIN
subsample of features.py (same 400 shots, seed 0) by NNLS, weights normalised to sum 1. Outputs: PCA bands 0-4, 4-16,
16-64 (psi-space least squares, the basis is orthonormal), q95, betaN. VALID is not used.

    .venv/bin/python paper/strat/fusion/stack.py   ->  runs/strat/fusion/stack_weights.json
Caveat recorded with the result: member predictions on TRAIN are in-sample (TRAIN residual RMS is 25-40 % below VALID).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import nnls

import fx


def main() -> None:
    sota, d = fx.data()
    tr = json.loads((fx.OUT / "features.json").read_text())["train_shots"]
    rows = {s: np.nonzero(d.shot == s)[0] for s in tr}
    mem = [fx.member_predict(fx.TRUNKS / m, d, rows) for m in fx.ENTRY]
    C = np.concatenate([np.asarray(d.C[rows[s], :64]) for s in tr]).astype(np.float64)
    L, H = sota.LABELS, sota.HEADS
    Sl = np.concatenate([d.S[rows[s]] for s in tr]).astype(np.float64)
    P = np.stack([np.concatenate([m[s][0] for s in tr]) for m in mem]).astype(np.float64)       # (M, N, 64)
    Q = np.stack([np.concatenate([m[s][1] for s in tr]) for m in mem]).astype(np.float64)       # (M, N, 5)
    w, info = {}, {}
    for band, (a, b) in {"0-4": (0, 4), "4-16": (4, 16), "16-64": (16, 64)}.items():
        A = P[:, :, a:b].reshape(len(mem), -1).T; y = C[:, a:b].ravel()
        x, _ = nnls(A, y); w[band] = (x / x.sum()).tolist()
    for nm, lab, h in (("q95", "efit_q95", "efit_q95"), ("betaN", "efit_beta_n", "efit_beta_n")):
        y = Sl[:, L.index(lab)]; ok = np.isfinite(y)
        x, _ = nnls(Q[:, ok, H.index(h)].T, y[ok]); w[nm] = (x / x.sum()).tolist()
    info = dict(members=fx.ENTRY, weights=w)
    (fx.OUT / "stack_weights.json").write_text(json.dumps(w, indent=1))
    (fx.OUT / "stack_info.json").write_text(json.dumps(info, indent=1))
    for k, v in w.items():
        print(f"[stack] {k:6s} " + " ".join(f"{x:.3f}" for x in v), flush=True)


if __name__ == "__main__":
    main()
