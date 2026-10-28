"""Doc 35 §42m diagnostic (label-free): do submitted public flux maps carry shot-2996-type axis failures? Per frame, the scorer-twin
O-point of the submitted psi vs the median of its +-5 neighbouring frames; reports frames / shots whose axis jumps by > thr.
    .venv/bin/python paper/strat/fusion/public_axis_scan.py runs/sophelio_sota/submission_F3/diii_d_public_test.npz
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import axis_shift as AX
from gpu_post import GPUPost


def main() -> None:
    z = np.load(sys.argv[1]); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    keys = sorted(k for k in z.files if k.endswith("_psirz")); out = {}
    for k in keys:
        psi = torch.tensor(z[k].astype(np.float64), device="cuda")
        out[k[:9]] = torch.cat([gp.axis(psi[i:i + 4096]) for i in range(0, len(psi), 4096)]).cpu().numpy()
    for thr in (0.05, 0.1, 0.2, 0.3):
        nf, bad = 0, []
        for s, a in out.items():
            n = len(a); med = np.stack([np.median(a[max(0, i - 5):i + 6], 0) for i in range(n)])
            m = np.abs(a - med).max(1) > thr; nf += int(m.sum())
            if m.any():
                bad.append((s, int(m.sum()), n, np.round(a[m][:3], 3).tolist()))
        print(f"[scan] thr {thr} m: {nf} frames in {len(bad)} shots of {len(out)}", flush=True)
        if thr >= 0.2:
            for b in bad[:15]:
                print("    ", b, flush=True)


if __name__ == "__main__":
    main()
