"""§37 routing strata (doc 35 §37, rule 2): generated VALIDATION instances for the 16 RouteFinder EAL-MD variants.

One stratum per problem variant (the unseen multi-depot variants of the EAL-MD row), equal size, seed 0. Instances are
drawn with RouteFinder's own generator (routefinder/data/generate.py: generate_mtvrp_data, the one that wrote the
official val (seed 4321) and test (seed 1234) files), so the distribution matches the benchmark; seed 0 differs from both.
The official test files are never opened. Writes runs/strat/routing/val_seed0/<variant>.npz and strata.json BEFORE any
strategy is scored.

    .venv-vrp/bin/python paper/strat/routing/make_strata.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "baseline" / "routefinder"))
from routefinder.data.generate import VARIANT_FEATURES, generate_mtvrp_data  # noqa: E402

OUT = ROOT / "runs" / "strat" / "routing"
SEED, PER_STRATUM, NUM_LOC, NUM_DEPOTS = 0, 128, 100, 3
# the 16 EAL-MD variants (run_eal.finetune_variant_names("md"): data dirs with "md" and without "mb")
BASE = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
        "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    d = OUT / "val_seed0"; d.mkdir(parents=True, exist_ok=True)
    strata = []
    for base in BASE:
        name = "md" + base.lower()
        np.random.seed(SEED)                      # same per-file seeding as RouteFinder's generate_dataset
        data = generate_mtvrp_data(dataset_size=PER_STRATUM, num_loc=NUM_LOC, num_depots=NUM_DEPOTS, variant=name)
        f = d / f"{name}.npz"; np.savez(f, **data)
        feats = VARIANT_FEATURES[base]
        strata.append(dict(variant=name, file=str(f.relative_to(ROOT)), sha256_16=sha(f), n=PER_STRATUM,
                           indices=list(range(PER_STRATUM)),
                           attributes=dict(multi_depot=True, open_route=feats["O"], time_windows=feats["TW"],
                                           duration_limit=feats["L"], backhaul=feats["B"], mixed_backhaul=feats["M"])))
    meta = dict(
        registered="doc 35 §37 (2026-09-27 22:43); strata written before any strategy is scored",
        written_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        split="generated VALIDATION instances (not the official val seed 4321, not the test seed 1234; test never loaded)",
        generator="baseline/routefinder/routefinder/data/generate.py:generate_mtvrp_data (numpy), np.random.seed(0) per file",
        num_loc=NUM_LOC, num_depots=NUM_DEPOTS, seed=SEED, per_stratum=PER_STRATUM, total=PER_STRATUM * len(BASE),
        stratum_key="problem variant (16 EAL-MD variants); attributes allow marginal pooling (O, TW, L, B); "
                    "mixed backhaul (M) is not part of the EAL-MD setting, so every stratum has M = False",
        metric="route cost (sum of travelled distance, RouteFinder reward = -cost); strategies compared as the paired "
               "relative cost change vs the §23 register row's protocol decode (dihedral x8, multi-start greedy)",
        strata=strata)
    (OUT / "strata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(strata)} strata x {PER_STRATUM} -> {OUT / 'strata.json'}")


if __name__ == "__main__":
    main()
