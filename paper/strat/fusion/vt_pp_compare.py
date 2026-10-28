"""§41y: paired comparison of two post-processing arms on all VALID + TEST shots (GPU twin parts from gpu_build.py --split vt).
    .venv/bin/python paper/strat/fusion/vt_pp_compare.py <base tag> <arm tag> [...]"""
from __future__ import annotations

import pickle
import sys

import numpy as np

import fx
import strategies as ST


def main() -> None:
    tags = sys.argv[1:]; _, nm = ST.load_entry(); P = {t: pickle.load(open(fx.OUT / f"parts_VT-{t}.pkl", "rb")) for t in tags}
    shots = sorted(P[tags[0]]); sb = np.stack([fx.shot_sums(P[tags[0]][s]) for s in shots]); cb = fx.composite(sb, nm)
    for t in tags[1:]:
        sn = np.stack([fx.shot_sums(P[t][s]) for s in shots]); c = fx.composite(sn, nm); bt = fx.paired_boot(sb, sn, nm)
        print(f"[vtpp] {t} vs {tags[0]}: S {c['S']:.6f} vs {cb['S']:.6f} delta {bt['delta']:+.6f} [{bt['lo']:+.6f}, {bt['hi']:+.6f}] | "
              + " ".join(f"{k} {c[k] - cb[k]:+.6f}" for k in ("r2_psi", "r2_qb", "dlcfs", "consistency")), flush=True)
        if "cons_each" in c:
            print("   per scalar: " + " ".join(f"{k} {c['cons_each'][k] - cb['cons_each'][k]:+.5f}" for k in c["cons_each"]), flush=True)


if __name__ == "__main__":
    main()
