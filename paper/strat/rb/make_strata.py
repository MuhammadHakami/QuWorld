"""§37 rayleigh_benard: fix the stratified VALID sample before any strategy is scored (doc 35 §37 rule 2).

Strata = Rayleigh (5) x Prandtl (7) x window position (onset t0 in [3, 9] | developed t0 in [10, 198]) = 70 strata.
Equal-size samples per stratum (N_PER windows, numpy seed 0, without replacement). VALID only; TEST is never opened.
Population weights (for the population-weighted overall mean, which estimates the full-VALID mean) are recorded too:
each (Ra, Pr) cell has 5 VALID trajectories, so cells weigh 1/35; inside a cell onset weighs 7/196, developed 189/196.

    .venv/bin/python paper/strat/rb/make_strata.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
import well2d  # noqa: E402

OUT = ROOT / "runs" / "strat" / "rb" / "strata.json"
N_PER = 20
ONSET_MAX_T0 = 9                      # onset: t0 < 10 (the §35 / app:alloc definition)
SEED = 0


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: strata are fixed once (delete by hand only if nothing was scored)")
    sp = well2d.Split("rayleigh_benard", "valid")
    params = []
    for f in sp.files:
        with h5py.File(f) as h:
            params.append((float(h.attrs["Rayleigh"]), float(h.attrs["Prandtl"])))
    t_lo, t_hi = well2d.N_HIST - 1, sp.T - 2                 # valid t0 range [3, 198]
    cells = {}
    for traj, (fi, _) in enumerate(sp.index):
        ra, pr = params[fi]
        for t0 in range(t_lo, t_hi + 1):
            pos = "onset" if t0 <= ONSET_MAX_T0 else "developed"
            cells.setdefault((ra, pr, pos), []).append((traj, t0))
    rng = np.random.default_rng(SEED)
    strata, n_pop = [], sum(len(v) for v in cells.values())
    for key in sorted(cells):
        pool = cells[key]
        pick = rng.choice(len(pool), N_PER, replace=False)
        ra, pr, pos = key
        strata.append({"rayleigh": ra, "prandtl": pr, "position": pos, "n_population": len(pool),
                       "pop_weight": len(pool) / n_pop,
                       "windows": [[int(pool[i][0]), int(pool[i][1])] for i in sorted(pick)]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "benchmark": "rayleigh_benard", "split": "valid", "seed": SEED, "n_per_stratum": N_PER,
        "strata_keys": ["rayleigh", "prandtl", "position"], "onset_t0_max": ONSET_MAX_T0,
        "t0_range": [t_lo, t_hi], "n_valid_trajectories": sp.n, "n_population_windows": n_pop,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "strata": strata}, indent=1))
    print(f"wrote {OUT}: {len(strata)} strata x {N_PER} = {len(strata) * N_PER} windows (population {n_pop})")


if __name__ == "__main__":
    main()
