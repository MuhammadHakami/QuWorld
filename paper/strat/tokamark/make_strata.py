"""§37 TokaMark: fix the strata and the equal-size VALID sample (seed 0) BEFORE any model is scored.

    .venv-tokamark/bin/python paper/strat/tokamark/make_strata.py  ->  runs/strat/tokamark/strata.json

Shot strata use inputs only (summary/ip via shot_meta.py): shot duration tercile (VALID tertiles) x termination class
(term_drop = fastest 5 ms fall of |Ip| after the peak, as a fraction of Ip max; 'fast' = at or above the VALID median).
Signal groups partition the 15 EFIT outputs by physics. Within-shot phase (ramp-up / flat-top / ramp-down, from Ip)
is a window-level breakdown stored in shot_meta.json. Every shot stratum gets the same number of sampled shots.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "strat" / "tokamark"
N_PER_STRATUM_CAP = 100

SIGNAL_GROUPS = {
    "shape": ["equilibrium-elongation", "equilibrium-elongation_axis", "equilibrium-triangularity_upper",
              "equilibrium-triangularity_lower", "equilibrium-minor_radius"],
    "position": ["equilibrium-x_point_r", "equilibrium-x_point_z", "equilibrium-magnetic_axis_r",
                 "equilibrium-magnetic_axis_z"],
    "pressure": ["equilibrium-beta_tor", "equilibrium-beta_pol", "equilibrium-beta_normal"],
    "field_q": ["equilibrium-q95", "equilibrium-bvac_rmag", "equilibrium-bphi_rmag"],
}


def main() -> None:
    meta = json.loads((OUT / "shot_meta.json").read_text())
    va = {int(k): v for k, v in meta["valid"].items() if "error" not in v}
    shots = np.array(sorted(va))
    dur = np.array([va[s]["duration"] for s in shots])
    term = np.array([va[s]["term_drop"] for s in shots])
    d_cut = np.quantile(dur, [1 / 3, 2 / 3]).tolist()
    t_cut = float(np.median(term))
    d_lab = np.digitize(dur, d_cut)                       # 0 short, 1 medium, 2 long
    t_lab = (term >= t_cut).astype(int)                   # 0 slow, 1 fast termination
    names = [f"{['short', 'medium', 'long'][d]}_{['slowterm', 'fastterm'][t]}" for d, t in zip(d_lab, t_lab)]
    cells = {n: shots[np.array(names) == n].tolist() for n in sorted(set(names))}
    n_per = int(min(N_PER_STRATUM_CAP, min(len(v) for v in cells.values())))
    rng = np.random.default_rng(0)
    sample = {n: sorted(int(s) for s in rng.choice(v, size=n_per, replace=False)) for n, v in sorted(cells.items())}
    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": 0, "split": "VALID (random split val column)",
        "n_valid_shots_with_meta": int(len(shots)), "n_per_stratum": n_per,
        "shot_strata_definition": {"duration_tercile_cuts_s": d_cut, "term_drop_median": t_cut,
                                   "duration": "global_end - global_start (TokaMarkDataset clock)",
                                   "term_drop": "max_t>t_peak (|Ip|(t) - |Ip|(t+5ms)) / max|Ip|"},
        "cell_sizes": {n: len(v) for n, v in cells.items()},
        "signal_groups": SIGNAL_GROUPS,
        "phase_codes": {"0": "ramp_up", "1": "flat_top", "2": "ramp_down"},
        "strata_shots": sample,
        "valid_sample_shots": sorted(s for v in sample.values() for s in v),
    }
    (OUT / "strata.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_valid_shots_with_meta", "n_per_stratum", "cell_sizes",
                                          "shot_strata_definition")}, indent=1))


if __name__ == "__main__":
    main()
