"""Doc 35 §45m: sub-frame inputs ("subf" block; inputs only). Our magnetics block (mag_fix.py) samples each signal at the EFIT time and
13 lags by point interpolation; the raw signals are sampled every 0.5 ms (20 kHz on some shots) with ~1 kA of sample noise on F-coil
currents of ~100-200 kA, so the up-down coil imbalance that sets the vertical position is noisy in a point sample. Per EFIT frame t,
on each signal's native samples (Ip with the organizers' time fix, s = sign of the shot's mean Ip, I = s * Ip in MA; coil currents
as shipped, kA):
  m10 = mean over [t - 10, t + 10] ms; m5 - m10; m2 - m10; v(t) - m10 (point sample minus the window mean, v(t) as mag_fix);
  least-squares slope over +-5 and +-10 ms (per s); residual rms about the +-10 ms line;
for Ip and the 20 coil circuits (ECOILA, bcoil, F1A .. F9B) -> 21 x 7; plus the time since the last and to the next core / edge
Thomson pulse (capped at 100 ms) -> 4. Idea from a public competitor study (Anulum, docs/STUDY.md section 4.16, read only, no code).
Writes runs/strat/fusion/subf_all.npy (cache row order) and subf_<fold tag>.npy (test-fold frame order) + subf_names.txt.

    .venv/bin/python paper/strat/fusion/subf_feats.py --workers 8 [--n 3] [--fold-only]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import fold as FD  # noqa: E402,I001
import mag_fix as MF  # noqa: E402

sys.path.insert(0, str(MF.ROOT / "baseline" / "sophelio-starter"))
from data_fixes import fix_d3d_ip_times  # noqa: E402

COILS = ["ECOILA", "bcoil"] + [f"F{i}{ab}" for i in range(1, 10) for ab in "AB"]
SIGS = ["ip"] + COILS
PARTS = ["m10", "m5d", "m2d", "ptd", "sl5", "sl10", "rms10"]
COLS = (["efit_times", "magnetics_time", "magnetics_plasma_current_times", "magnetics_plasma_current", "thomson_core_times",
         "thomson_edge_times"] + [f"magnetics_{c}" for c in COILS])
NAMES = [f"sf_{s}_{p}" for s in SIGS for p in PARTS] + ["sf_th_core_since", "sf_th_core_to", "sf_th_edge_since", "sf_th_edge_to"]


def window_stats(tau: np.ndarray, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n_frames, 7) window statistics of samples (tau [ms], v) around each frame time t, via prefix sums (O(1) per window)."""
    ok = np.isfinite(tau) & np.isfinite(v); tau, v = tau[ok], v[ok]
    if len(tau) < 8:
        return np.zeros((len(t), 7))
    o = np.argsort(tau, kind="stable"); tau, v = tau[o], v[o]
    v0 = np.median(v); x = v - v0                                                # centred values keep the prefix sums well conditioned
    def pre(a):
        return np.r_[0.0, np.cumsum(a)]
    P = {k: pre(a) for k, a in (("n", np.ones_like(x)), ("x", x))}

    def mean(W):
        lo, hi = np.searchsorted(tau, t - W, "left"), np.searchsorted(tau, t + W, "right"); n = hi - lo
        m = (P["x"][hi] - P["x"][lo]) / np.maximum(n, 1)
        return np.where(n > 0, m, np.interp(t, tau, x)), lo, hi

    def line(W):
        lo, hi = np.searchsorted(tau, t - W, "left"), np.searchsorted(tau, t + W, "right"); out = np.zeros((len(t), 2))
        for i, (a, b) in enumerate(zip(lo, hi)):
            if b - a < 4:
                continue
            u = tau[a:b] - t[i]; y = x[a:b]; um, ym = u.mean(), y.mean(); du = u - um
            sl = (du * (y - ym)).sum() / max((du * du).sum(), 1e-12); r = y - ym - sl * du
            out[i] = sl * 1000.0, np.sqrt((r * r).mean())                          # slope per s; residual rms
        return out
    m10, _, _ = mean(10.0); m5, _, _ = mean(5.0); m2, _, _ = mean(2.0); pt = np.interp(t, tau, x)
    l5, l10 = line(5.0), line(10.0)
    return np.stack([m10 + v0, m5 - m10, m2 - m10, pt - m10, l5[:, 0], l10[:, 0], l10[:, 1]], 1)


def pulse_gaps(tp, t: np.ndarray) -> np.ndarray:
    tp = np.sort(np.asarray(tp, np.float64)); tp = tp[np.isfinite(tp)]
    if len(tp) == 0:
        return np.full((len(t), 2), 100.0)
    j = np.searchsorted(tp, t, "right")
    since = np.where(j > 0, t - tp[np.clip(j - 1, 0, len(tp) - 1)], 100.0)
    to = np.where(j < len(tp), tp[np.clip(j, 0, len(tp) - 1)] - t, 100.0)
    return np.clip(np.stack([since, to], 1), 0.0, 100.0)


def feats(row) -> np.ndarray:
    te_ = np.asarray(row["efit_times"], np.float64); mt = np.asarray(row["magnetics_time"], np.float64)
    ipt = np.asarray(fix_d3d_ip_times(row), np.float64); ip = np.asarray(row["magnetics_plasma_current"], np.float64)
    m = np.isfinite(ip) & np.isfinite(ipt)
    s = 1.0 if np.mean(np.interp(te_, ipt[m], ip[m])) >= 0 else -1.0
    out = [window_stats(ipt, s * ip / 1000.0, te_)]
    for c in COILS:
        out.append(window_stats(mt, np.asarray(row[f"magnetics_{c}"], np.float64), te_))
    out += [pulse_gaps(row["thomson_core_times"], te_), pulse_gaps(row["thomson_edge_times"], te_)]
    return np.concatenate(out, 1).astype(np.float32)


def _one(args):
    path, name = args
    row = pd.read_parquet(path / f"{name}.parquet", columns=COLS).iloc[0]
    return name, feats(row)


def build(src: pathlib.Path, names, counts, dst: pathlib.Path, workers: int) -> None:
    pos = dict(zip(names, np.r_[0, np.cumsum(counts)[:-1]])); H = np.zeros((int(np.sum(counts)), len(NAMES)), np.float32)
    with mp.Pool(workers) as pool:
        for i, (nm, f) in enumerate(pool.imap_unordered(_one, [(src, n) for n in names], chunksize=4)):
            assert len(f) == counts[names.index(nm)] if len(names) < 2000 else True
            s = pos[nm]; H[s:s + len(f)] = f
            if (i + 1) % 1000 == 0:
                print(f"[subf] {dst.name} {i + 1}/{len(names)}", flush=True)
    bad = ~np.isfinite(H)
    if bad.any():
        raise SystemExit(f"non-finite subf feature: {bad.sum()} values, columns {sorted(set(np.nonzero(bad)[1].tolist()))[:10]}")
    np.save(dst, H); print(f"[subf] {dst.name} {H.shape}", flush=True)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--fold-only", action="store_true"); a = ap.parse_args()
    meta = torch.load(MF.CACHE / "meta.pt", weights_only=False); names = list(meta["names"]); shot = np.load(MF.CACHE / "shot.npy")
    counts = np.bincount(shot, minlength=len(names))
    if a.n:                                                                        # smoke: first n shots, checks frame counts
        for nm, c in zip(names[:a.n], counts[:a.n]):
            _, f = _one((MF.RAW, nm)); assert len(f) == c, (nm, len(f), c)
            mid = len(f) // 2
            print(nm, f.shape, "mid frame", dict(zip(NAMES[:7] + NAMES[7 * 12:7 * 13] + NAMES[-4:],
                                                       np.round(np.r_[f[mid, :7], f[mid, 7 * 12:7 * 13], f[mid, -4:]], 3).tolist())))
        return
    if not a.fold_only:
        (MF.OUT / "subf_names.txt").write_text("\n".join(NAMES))
        build(MF.RAW, names, counts, MF.OUT / "subf_all.npy", a.workers)
    proc = MF.ROOT / "data" / "sophelio_proc" / FD.FOLD; pn = [f.stem for f in sorted(proc.glob("*.npz"))]
    pc = np.asarray([len(np.load(proc / f"{nm}.npz")["t"]) for nm in pn])
    build(MF.RAW.parent / FD.FOLD, pn, pc, MF.OUT / f"subf_{FD.TAG}.npy", a.workers)
    print("SUBF_DONE", flush=True)


if __name__ == "__main__":
    main()
