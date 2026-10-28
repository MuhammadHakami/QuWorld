"""Fusion plan B1a: true LCFS contours (train parquet "bonus context": efit_lcfs_r / efit_lcfs_z / efit_lcfs_n) resampled to 32
arc-length points per frame, rows aligned with the cache (data/sophelio_proc/cache_k256: shots in meta["names"] order, frames
in efit_times order). Output cache_k256/lcfs32.npy (N, 32, 2) float32 (R, Z), NaN where a frame has no contour.

    .venv/bin/python paper/strat/fusion/lcfs_extract.py --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data" / "diii_d_train"
CACHE = ROOT / "data" / "sophelio_proc" / "cache_k256"
K = 32


def resample(r, z, k=K):
    """Closed contour -> k points equally spaced in arc length (starting at the lowest point, the lower X-point side)."""
    r, z = np.asarray(r, np.float64), np.asarray(z, np.float64); m = np.isfinite(r) & np.isfinite(z); r, z = r[m], z[m]
    if len(r) < 4:
        return np.full((k, 2), np.nan, np.float32)
    i0 = int(np.argmin(z)); r, z = np.roll(r, -i0), np.roll(z, -i0)
    rr, zz = np.r_[r, r[0]], np.r_[z, z[0]]; s = np.r_[0, np.cumsum(np.hypot(np.diff(rr), np.diff(zz)))]
    if s[-1] <= 0:
        return np.full((k, 2), np.nan, np.float32)
    q = np.linspace(0, s[-1], k, endpoint=False)
    return np.stack([np.interp(q, s, rr), np.interp(q, s, zz)], 1).astype(np.float32)


def one(name):
    row = pd.read_parquet(SRC / f"{name}.parquet", columns=["efit_times", "efit_lcfs_r", "efit_lcfs_z", "efit_lcfs_n"]).iloc[0]
    T = len(np.asarray(row["efit_times"])); out = np.full((T, K, 2), np.nan, np.float32)
    R, Z, N = row["efit_lcfs_r"], row["efit_lcfs_z"], np.asarray(row["efit_lcfs_n"]).ravel()
    for t in range(T):
        n = int(N[t]) if t < len(N) and np.isfinite(N[t]) else 0
        if n > 3:
            out[t] = resample(np.asarray(R[t])[:n], np.asarray(Z[t])[:n])
    return name, out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    meta = torch.load(CACHE / "meta.pt", weights_only=False); names = list(meta["names"]); shot = np.load(CACHE / "shot.npy")
    counts = np.bincount(shot, minlength=len(names)); res = {}
    with mp.Pool(a.workers) as pool:
        for i, (nm, arr) in enumerate(pool.imap_unordered(one, names, chunksize=8)):
            res[nm] = arr
            if (i + 1) % 1000 == 0:
                print(f"[lcfs] {i + 1}/{len(names)}", flush=True)
    bad = [nm for j, nm in enumerate(names) if len(res[nm]) != counts[j]]
    assert not bad, f"frame-count mismatch on {len(bad)} shots, e.g. {bad[:3]}"
    out = np.concatenate([res[nm] for nm in names]); np.save(CACHE / "lcfs32.npy", out)
    print(f"LCFS_DONE {out.shape} frames with contour {np.isfinite(out[:, 0, 0]).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
