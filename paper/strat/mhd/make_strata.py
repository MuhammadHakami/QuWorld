"""Doc 35 §37, MHD_64: fix the stratified TRAIN and VALID samples (seed 0) before any strategy is scored.

Strata = Alfven-Mach regime (file label Ma 0.7 / 2; equal to the causal guide-field read, §36 step 0) x time position of
the last input frame t0 (4 equal bins over t0 = 3..98). Equal-size samples per stratum:
  VALID: every trajectory (10 = 2 Ma x 5 Ms, one per file) x 4 time bins x 4 windows -> 20 per stratum, 160 in all.
  TRAIN: every file (10) x 4 time bins x 3 windows, trajectory drawn uniformly within the file -> 15 per stratum, 120.
TEST is never opened (only TRAIN/VALID file names and shapes are read). Writes runs/strat/mhd/strata.json.

    uv run --no-sync python paper/strat/mhd/make_strata.py
"""
from __future__ import annotations

import datetime
import glob
import json
import pathlib
import re

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "strat" / "mhd" / "strata.json"
N_HIST, T = 4, 100
T0 = np.arange(N_HIST - 1, T - 1)                         # last input frame 3..98 (target t0 + 1 <= 99)
BINS = np.array_split(T0, 4)
K_VALID, K_TRAIN = 4, 3


def files(split: str) -> list[dict]:
    """Sorted like the_well / cache.WellND, so the global trajectory index matches WellND(split).index."""
    out, g = [], 0
    for f in sorted(glob.glob(str(ROOT / "data/well/MHD_64/data" / split / "*.hdf5"))):
        with h5py.File(f) as h:
            n = h["t0_fields/density"].shape[0]
        ma, ms = map(float, re.search(r"Ma_([0-9.]+)_Ms_([0-9.]+)\.hdf5", f).groups())
        out.append({"file": pathlib.Path(f).name, "Ma": ma, "Ms": ms, "first": g, "n": n}); g += n
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    res = {"protocol": "doc 35 §37 (exploratory; TEST never loaded)", "seed": 0,
           "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "strata": "Ma regime (0.7 | 2) x t0 bin (4 equal bins over t0 = 3..98)",
           "time_bins": [[int(b[0]), int(b[-1])] for b in BINS], "samples": {}}
    for split, k in (("valid", K_VALID), ("train", K_TRAIN)):
        fl, rows = files(split), []
        for f in fl:
            for bi, b in enumerate(BINS):
                t0s = rng.choice(b, k, replace=False)
                for t0 in sorted(int(v) for v in t0s):
                    traj = f["first"] + int(rng.integers(f["n"]))
                    rows.append({"traj": traj, "t0": t0, "Ma": f["Ma"], "Ms": f["Ms"], "tbin": bi, "file": f["file"]})
        res["samples"][split] = rows
        res[f"files_{split}"] = fl
        cnt = {}
        for r in rows:
            cnt[f"Ma{r['Ma']}_b{r['tbin']}"] = cnt.get(f"Ma{r['Ma']}_b{r['tbin']}", 0) + 1
        res[f"counts_{split}"] = cnt
        print(split, len(rows), cnt)
    assert not OUT.exists(), f"{OUT} exists: strata are fixed once"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
