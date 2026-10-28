"""Fusion §41q fast-proxy read-out: mean consistency-scalar R2 (VALID sample rows, pooled variance) of head_boundary_v8_P<arm>_s<seed>
per arm and seed, and the paired delta vs the control arm (same seed, same subset).

    .venv/bin/python paper/strat/fusion/proxy_report.py [ctrl thz canon hfull aux cs]
"""
from __future__ import annotations

import sys

import numpy as np

import fx  # noqa: F401,I001
import os
PFX = os.environ.get("PROXY_PFX", "P")


def score(P, Y):
    r = []
    for j in range(7):
        ok = np.isfinite(Y[:, j]); r.append(1 - np.mean((Y[ok, j] - P[ok, j]) ** 2) / np.var(Y[ok, j]))
    return float(np.mean(r)), r


def main() -> None:
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); Y = lab["y_valid"].astype(np.float64)
    arms = sys.argv[1:] or ["ctrl", "thz", "canon", "hfull", "aux", "cs"]; res = {}
    for a in arms:
        for s in (1, 2, 3, 4, 5, 6, 7, 8):
            f = fx.OUT / f"head_boundary_v8_{PFX}{a}_s{s}.npz"
            if f.exists():
                res[(a, s)] = score(np.load(f)["pred_valid"].astype(np.float64), Y)
    for a in arms:
        ss = [s for s in (1, 2, 3, 4, 5, 6, 7, 8) if (a, s) in res]
        if not ss:
            continue
        m = np.mean([res[(a, s)][0] for s in ss]); d = [res[(a, s)][0] - res[("ctrl", s)][0] for s in ss if ("ctrl", s) in res]
        print(f"[proxy] {a:6s} seeds {ss} mean {m:.5f} | paired delta vs ctrl {np.round(d, 5).tolist()} mean {np.mean(d) if d else float('nan'):+.5f}", flush=True)


if __name__ == "__main__":
    main()
