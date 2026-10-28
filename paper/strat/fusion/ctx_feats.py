"""Fusion §41q shot-context inputs (inputs only; the whole shot's inputs are given at prediction time). Head errors are 30-55% per-shot
bias (R_axis, Z_axis, li, volume), so every frame gets:
  ctx  : the shot means of the 21 corrected current signals (lag 0) and of the 105-column fixed-coordinate Thomson block (126 cols)
  thzs : the fixed-coordinate Thomson block averaged over the frame and its +-2 neighbours in the same shot (105 cols)
Writes runs/strat/fusion/{ctx,thzs}_{all,public}.npy (cache / public row order).

    .venv/bin/python paper/strat/fusion/ctx_feats.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import fold as FD  # noqa: E402,I001

ROOT = pathlib.Path(__file__).resolve().parents[3]; OUT = ROOT / "runs" / "strat" / "fusion"; PROC = ROOT / "data" / "sophelio_proc"


def build(mag, thz, shot, tag):
    st = np.r_[0, np.nonzero(np.diff(shot))[0] + 1]; n = np.diff(np.r_[st, len(shot)]); assert len(np.unique(shot[st])) == len(st)
    cols = json.loads((OUT / "magfix.json").read_text())["columns"]; sig = [i for i, c in enumerate(cols) if "_lag" not in c]
    M = np.asarray(mag[:, sig], np.float64); T = np.asarray(thz, np.float64)
    ctx = np.concatenate([np.repeat(np.add.reduceat(M, st) / n[:, None], n, 0), np.repeat(np.add.reduceat(T, st) / n[:, None], n, 0)], 1)
    cs = np.vstack([np.zeros((1, T.shape[1])), np.cumsum(T, 0)]); idx = np.arange(len(shot)); s0 = np.repeat(st, n); s1 = s0 + np.repeat(n, n)
    lo, hi = np.maximum(idx - 2, s0), np.minimum(idx + 3, s1); thzs = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    np.save(OUT / f"ctx_{tag}.npy", ctx.astype(np.float32)); np.save(OUT / f"thzs_{tag}.npy", thzs.astype(np.float32))
    print(f"[ctx] {tag}: ctx {ctx.shape} thzs {thzs.shape} finite {np.isfinite(ctx).all() and np.isfinite(thzs).all()}", flush=True)


def main() -> None:
    shot = np.load(PROC / "cache_k256" / "shot.npy")
    build(np.load(OUT / "X_magfix_all.npy", mmap_mode="r"), np.load(PROC / "cache_k256" / "thz.npy", mmap_mode="r"), shot, "all")
    proc = PROC / FD.FOLD; pn = sorted(proc.glob("*.npz")); pc = [len(np.load(f)["t"]) for f in pn]
    build(np.load(OUT / f"X_magfix_{FD.TAG}.npy", mmap_mode="r"), np.load(PROC / f"thz_{FD.TAG}.npy", mmap_mode="r"), np.repeat(np.arange(len(pc)), pc), "public")
    print("CTX_DONE", flush=True)


if __name__ == "__main__":
    main()
