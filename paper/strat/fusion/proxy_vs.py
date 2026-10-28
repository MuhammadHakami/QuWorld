"""Fusion §41q proxy read-out of arms against a chosen base arm (paired by seed): proxy_vs.py <base> <arm> [<arm> ...]."""
from __future__ import annotations

import sys

import numpy as np

import fx  # noqa: F401,I001
import proxy_report as PR


def main() -> None:
    Y = np.load(fx.OUT / "boundary_labels_v8.npz")["y_valid"].astype(np.float64); base, arms = sys.argv[1], sys.argv[2:]
    def sc(a, s):
        f = fx.OUT / f"head_boundary_v8_P2{a}_s{s}.npz"
        return PR.score(np.load(f)["pred_valid"].astype(np.float64), Y)[0] if f.exists() else None
    for a in arms:
        d = [sc(a, s) - sc(base, s) for s in range(1, 11) if sc(a, s) is not None and sc(base, s) is not None]
        print(f"[proxy] {a:7s} vs {base}: seeds {len(d)} deltas {np.round(d, 5).tolist()} mean {np.mean(d) if d else float('nan'):+.5f}", flush=True)


if __name__ == "__main__":
    main()
