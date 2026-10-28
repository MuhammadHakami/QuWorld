"""Doc 35 §41c: long-window history features for the Fusion trunk (inputs only, never labels).

The trunk sees the coil currents and Ip at the frame and +-200 ms lags, plus Thomson summaries. Internal inductance li,
q95 and the flux shape depend on how the current profile evolved over seconds (resistive diffusion), so every frame gets,
from the frames of its own shot (20 ms spacing; limited phases are absent from the frame list):
- causal means over [t - W, t] for W = 0.5, 1, 2 s and the mean since the shot's first frame,
- anti-causal means over [t, t + W] for W = 0.5, 1 s (the whole shot's inputs are given at prediction time),
of each current magnetics signal (21) and the Thomson core / edge Te and ne means (4), each as a difference from the
frame's own value (so the trunk reads the history as a trend), plus the elapsed time since the first frame.
Writes <cache>/hist.npy (float32, rows aligned with X.npy) and hist_names.txt.

    .venv/bin/python paper/strat/fusion/hist_feats.py
"""
from __future__ import annotations

import sys

import numpy as np

import fx  # noqa: F401,I001  (paths)
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

CAUSAL = (500.0, 1000.0, 2000.0, np.inf)
ANTI = (500.0, 1000.0)


def window_means(t, V, W, causal):
    """Mean of V over the frames with t' in [t - W, t] (causal) or [t, t + W] (anti-causal), per frame of one shot."""
    cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, 0)])
    if causal:
        j0 = np.searchsorted(t, t - W, side="left"); j1 = np.arange(1, len(t) + 1)
    else:
        j0 = np.arange(len(t)); j1 = np.searchsorted(t, t + W, side="right")
    return (cs[j1] - cs[j0]) / np.maximum(j1 - j0, 1)[:, None]


def main() -> None:
    fn = open(fx.ROOT / "data" / "sophelio_proc" / "feature_names.txt").read().split()
    X = np.load(sota.CACHE / "X.npy", mmap_mode="r"); shot = np.load(sota.CACHE / "shot.npy")
    sig = [i for i, n in enumerate(fn) if n.startswith("mag_") and "_lag" not in n]
    sig += [fn.index(n) for n in ("thomson_core_Te_mean", "thomson_core_ne_mean", "thomson_edge_Te_mean", "thomson_edge_ne_mean") if n in fn]
    it = fn.index("t"); starts = np.r_[0, np.nonzero(np.diff(shot))[0] + 1, len(shot)]
    names = [f"h{'c' if c else 'a'}{int(W) if np.isfinite(W) else 'all'}_{fn[i]}" for c, Ws in ((True, CAUSAL), (False, ANTI)) for W in Ws for i in sig]
    names.append("h_elapsed")
    H = np.zeros((len(shot), len(names)), np.float32)
    for a, b in zip(starts[:-1], starts[1:]):
        t = np.asarray(X[a:b, it], np.float64); V = np.nan_to_num(np.asarray(X[a:b][:, sig], np.float64)); cols = []
        for c, Ws in ((True, CAUSAL), (False, ANTI)):
            for W in Ws:
                cols.append(window_means(t, V, W, c) - V)
        H[a:b, :-1] = np.concatenate(cols, 1); H[a:b, -1] = t - t[0]
    np.save(sota.CACHE / "hist.npy", H); (sota.CACHE / "hist_names.txt").write_text("\n".join(names))
    print(f"[hist] {H.shape} features from {len(sig)} signals; finite {np.isfinite(H).all()}", flush=True)


if __name__ == "__main__":
    main()
