"""Fusion §41q pre-upload sanity gate for builds that cannot be validated locally: agreement of a new public npz with a reference
public npz (same keys): psi agreement R2 (pooled), q95 / betaN agreement R2, all finite. Exit 1 if psi agreement < --min.

    .venv/bin/python paper/strat/fusion/agree.py <new.npz> <ref.npz> [--min 0.999]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("new"); ap.add_argument("ref"); ap.add_argument("--min", type=float, default=0.999); a = ap.parse_args()
    A, B = np.load(a.new), np.load(a.ref); keys = sorted(k for k in B.files if k.endswith("_psirz")); assert set(A.files) == set(B.files), "key sets differ"
    acc = {"psi": [0.0, 0.0, 0.0, 0], "q95": [0.0, 0.0, 0.0, 0], "betaN": [0.0, 0.0, 0.0, 0]}
    for k in keys:
        for q, kk in (("psi", k), ("q95", k.replace("_psirz", "_q95")), ("betaN", k.replace("_psirz", "_betaN"))):
            x, y = np.asarray(A[kk], np.float64).ravel(), np.asarray(B[kk], np.float64).ravel()
            assert np.isfinite(x).all(), kk
            s = acc[q]; s[0] += ((x - y) ** 2).sum(); s[1] += y.sum(); s[2] += (y ** 2).sum(); s[3] += y.size
    out = {}
    for q, (res, sm, sq, n) in acc.items():
        out[q] = 1 - res / (sq - sm * sm / n)
    print("[agree] " + " ".join(f"{q} {v:.6f}" for q, v in out.items()), flush=True)
    sys.exit(0 if out["psi"] >= a.min else 1)


if __name__ == "__main__":
    main()
