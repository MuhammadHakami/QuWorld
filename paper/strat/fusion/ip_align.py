"""§37 fusion: is the plasma-current input aligned with the EFIT frames? (TRAIN shots only, raw parquet inputs.)

    .venv/bin/python paper/strat/fusion/ip_align.py [--n 60]   ->  runs/strat/fusion/ip_align.json

All shots carry the same `magnetics_plasma_current_times` vector (-858.19 ... 14500.81 ms, 30,719 samples). For each
shot we find the time shift tau (claimed Ip time = true time + tau) that maximises the correlation between dIp/dt and
-dI_E/dt (the Ohmic E-coil drives the plasma current; both are inputs), on a 1 ms grid. 2 kHz shots are the control
(expected tau ~ 0). Inputs only; no label is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import fx

COLS = ["efit_times", "magnetics_plasma_current_times", "magnetics_plasma_current", "magnetics_time", "magnetics_ECOILA"]


def split_of(name: str) -> str:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else ("val" if h < 10 else "train")


def best_shift(row, taus=np.arange(-1000, 5000, 2.0)) -> tuple[float, float, float]:
    ipt = np.asarray(row["magnetics_plasma_current_times"], float); ip = np.asarray(row["magnetics_plasma_current"], float)
    mt = np.asarray(row["magnetics_time"], float); ec = np.asarray(row["magnetics_ECOILA"], float)
    m = np.isfinite(ip); me = np.isfinite(ec)
    g = np.arange(-500.0, 8000.0, 2.0)                               # true-time grid (ms)
    e = np.interp(g, mt[me], ec[me]); de = -np.gradient(e)
    de = np.convolve(de, np.ones(5) / 5, "same")
    best, bc, c0 = 0.0, -2.0, np.nan
    for tau in taus:
        x = np.interp(g + tau, ipt[m], ip[m]); dx = np.convolve(np.gradient(x), np.ones(5) / 5, "same")
        if dx.std() == 0 or de.std() == 0:
            continue
        c = abs(np.corrcoef(dx, de)[0, 1])
        if tau == 0:
            c0 = c
        if c > bc:
            bc, best = c, tau
    return float(best), float(bc), float(c0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    root = fx.ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data" / "diii_d_train"
    files = sorted(root.glob("*.parquet")); rng = np.random.default_rng(0)
    pick = [f for f in rng.permutation(files) if split_of(f.stem) == "train"][:a.n]
    rows = []
    for f in pick:
        r = pd.read_parquet(f, columns=COLS).iloc[0]
        tau, c, c0 = best_shift(r)
        nmag = len(r["magnetics_time"]); t0 = float(np.asarray(r["magnetics_time"])[0])
        rows.append(dict(shot=f.stem, nmag=nmag, mag_t0=t0, tau=tau, corr=c, corr_at0=c0))
        print(f"{f.stem} nmag {nmag:6d} mag_t0 {t0:9.1f} tau {tau:7.1f} corr {c:.3f} corr@0 {c0:.3f} "
              f"tau+mag_t0 {tau + t0:8.1f}", flush=True)
    fast = [r for r in rows if r["nmag"] > 200000]; slow = [r for r in rows if r["nmag"] <= 200000]
    summ = {k: dict(n=len(v), tau_median=float(np.median([r["tau"] for r in v])) if v else None,
                    tau_iqr=[float(np.percentile([r["tau"] for r in v], q)) for q in (25, 75)] if v else None,
                    tau_plus_magt0_median=float(np.median([r["tau"] + r["mag_t0"] for r in v])) if v else None,
                    tau_plus_magt0_iqr=[float(np.percentile([r["tau"] + r["mag_t0"] for r in v], q)) for q in (25, 75)] if v else None)
            for k, v in (("20kHz", fast), ("2kHz", slow))}
    (fx.OUT / "ip_align.json").write_text(json.dumps(dict(summary=summ, shots=rows), indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
