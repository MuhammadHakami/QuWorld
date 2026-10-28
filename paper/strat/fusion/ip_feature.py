"""§37 fusion: re-aligned plasma-current input for every TRAIN and VALID frame of the cache (TEST shots are skipped,
never read). Rule from ip_align.py (inputs only, 40 TRAIN shots): the Ip samples start at magnetics_time[0] - 2.25 ms
on 20 kHz shots (tau + mag_t0 = -855.9 ms, IQR 2.7 ms) and at magnetics_time[0] + 8191.6 ms on 2 kHz shots
(tau + mag_t0 = -9049.8 ms, IQR 13 ms), instead of the shipped shared vector -858.19 + 0.5 i ms.

    .venv/bin/python paper/strat/fusion/ip_feature.py   ->  runs/strat/fusion/ip_fixed.npy  (N_frames, 6), NaN on TEST rows

Columns: Ip(t), Ip(t-200 ms), Ip(t-50 ms), Ip(t+20 ms), dIp/dt (kA/ms, +-5 ms), Ip_shipped(t) (the cache's own column,
for the check). Times are true milliseconds (the cache's lag columns used l/1000, i.e. microseconds; see report).
"""
from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pandas as pd

import fx

RAW = fx.ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data" / "diii_d_train"
OFF_20K, OFF_2K = -2.25, 8191.6


def one(name: str):
    r = pd.read_parquet(RAW / f"{name}.parquet",
                        columns=["efit_times", "magnetics_plasma_current", "magnetics_plasma_current_times", "magnetics_time"]).iloc[0]
    t = np.asarray(r["efit_times"], float); ip = np.asarray(r["magnetics_plasma_current"], float)
    mt = np.asarray(r["magnetics_time"], float); ipt_ship = np.asarray(r["magnetics_plasma_current_times"], float)
    m = np.isfinite(ip)
    start = mt[0] + (OFF_20K if len(mt) > 200000 else OFF_2K)
    ipt = start + 0.5 * np.arange(len(ip))
    f = lambda q: np.interp(q, ipt[m], ip[m])  # noqa: E731
    out = np.stack([f(t), f(t - 200), f(t - 50), f(t + 20), (f(t + 5) - f(t - 5)) / 10.0,
                    np.interp(t, ipt_ship[m], ip[m])], 1).astype(np.float32)
    return name, out, len(mt)


def main() -> None:
    sota, d = fx.data()
    names = d.meta["names"]
    todo = [j for j in range(len(names)) if sota.split_of(names[j]) != "test"]
    starts = np.r_[0, np.nonzero(np.diff(d.shot))[0] + 1]
    assert (d.shot[starts] == np.arange(len(names))).all()
    ends = np.r_[starts[1:], len(d.shot)]
    res = np.full((len(d.shot), 6), np.nan, np.float32)
    nmag = {}
    with mp.get_context("spawn").Pool(4) as pool:
        for k, (name, out, nm) in enumerate(pool.imap_unordered(one, [names[j] for j in todo], chunksize=8)):
            nmag[name] = (out, nm)
            if (k + 1) % 1000 == 0:
                print(f"[ip] {k + 1}/{len(todo)}", flush=True)
    idx = {n: j for j, n in enumerate(names)}
    tcol = d.fn.index("t"); ipcol = d.fn.index("mag_plasma_current")
    bad = 0
    for name, (out, nm) in nmag.items():
        j = idx[name]; a, b = starts[j], ends[j]
        if b - a != len(out):
            bad += 1
            continue
        res[a:b] = out
    np.save(fx.OUT / "ip_fixed.npy", res)
    ok = np.isfinite(res[:, 0]) & (d.split != 2)
    fr_live_fixed = float((np.abs(res[ok, 0]) > 150).mean()); fr_live_ship = float((np.abs(res[ok, 5]) > 150).mean())
    is20 = {n: v[1] > 200000 for n, v in nmag.items()}
    s = dict(shots=len(nmag), length_mismatch=bad, frac_frames_absIp_gt150kA_fixed=fr_live_fixed,
             frac_frames_absIp_gt150kA_shipped=fr_live_ship, frac_shots_20kHz=float(np.mean(list(is20.values()))))
    xs = np.asarray(d.X[:, ipcol]); sel = np.nonzero(ok)[0]
    s["max_abs_diff_shipped_vs_cache_column"] = float(np.nanmax(np.abs(xs[sel] - res[sel, 5])))
    (fx.OUT / "ip_fixed.json").write_text(json.dumps(s, indent=1))
    print(s)


if __name__ == "__main__":
    main()
