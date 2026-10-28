"""§37 fusion step 0: fix the strata and the equal-size stratified VALID sample BEFORE any strategy is scored.

    .venv/bin/python paper/strat/fusion/strata.py   ->  runs/strat/fusion/strata.json

Shot strata (VALID only): plasma-current regime (terciles of the shot-median |Ip|, an input) x shape class (terciles of
the shot-median lower-minus-upper triangularity of the ground-truth equilibrium, from the scorer's cached VALID
reference: lower-biased / balanced / upper-biased). Equal-size sample of n shots per cell, numpy seed 0. Frame strata
inside a shot: ramp-up (elapsed fraction < 0.1), flat-top, ramp-down (> 0.9), from the frame times (inputs). Output
groups: psi, q95/betaN, D_LCFS, the seven consistency scalars. VALID shots outside the sample form the SELECTION set
(any VALID-based choice of a strategy's hyper-parameter is made there, never on the scored sample).
"""
from __future__ import annotations

import json

import numpy as np

import fx


def main() -> None:
    _, d = fx.data()
    valid = d.shots_of(1)
    ip_col = d.fn.index("mag_plasma_current")
    ip, asym, kap, nfr = [], [], [], []
    for s in valid:
        rows = fx.valid_rows(d, s)
        ip.append(float(np.median(np.abs(np.asarray(d.X[rows, ip_col])))))
        z = np.load(fx.REF / f"ref_{d.meta['names'][s]}.npz", allow_pickle=True)
        c = np.where(z["cmask"], z["cons"], np.nan)
        asym.append(float(np.nanmedian(c[:, 4] - c[:, 3]))); kap.append(float(np.nanmedian(c[:, 2]))); nfr.append(len(rows))
    ip, asym = np.array(ip), np.array(asym)
    qi = np.quantile(ip, [1 / 3, 2 / 3]); qa = np.quantile(asym, [1 / 3, 2 / 3])
    ipc = np.digitize(ip, qi); shc = np.digitize(asym, qa)
    cell = ipc * 3 + shc
    sizes = np.bincount(cell, minlength=9)
    n = int(min(20, sizes.min()))
    rng = np.random.default_rng(0)
    sample = {}
    for c in range(9):
        members = [valid[i] for i in np.nonzero(cell == c)[0]]
        sample[c] = sorted(int(x) for x in rng.choice(members, size=n, replace=False))
    in_sample = {s for v in sample.values() for s in v}
    names_ip = ["Ip-low", "Ip-mid", "Ip-high"]; names_sh = ["lower-biased", "balanced", "upper-biased"]
    out = dict(
        created="registered before any strategy score (§37 rule 2)", seed=0, n_per_stratum=n,
        ip_tercile_edges_kA=qi.tolist(), tri_asym_tercile_edges=qa.tolist(),
        strata={f"{names_ip[c // 3]}|{names_sh[c % 3]}": dict(cell=c, n_valid=int(sizes[c]), shots=sample[c],
                                                              shot_names=[d.meta["names"][s] for s in sample[c]])
                for c in range(9)},
        frame_strata={"ramp-up": "elapsed fraction < 0.1", "flat-top": "0.1 <= fraction <= 0.9", "ramp-down": "> 0.9"},
        output_groups=["psi", "q95", "betaN", "dlcfs"] + fx.CONS,
        selection_shots=[int(s) for s in valid if s not in in_sample],
        valid_shot_info={int(s): dict(ip_kA=float(ip[i]), tri_asym=float(asym[i]), kappa=float(kap[i]), frames=int(nfr[i]),
                                      cell=int(cell[i])) for i, s in enumerate(valid)})
    (fx.OUT / "strata.json").write_text(json.dumps(out, indent=1))
    print(f"[strata] cells {sizes.tolist()} n/stratum {n} sample {len(in_sample)} selection {len(out['selection_shots'])}")
    print(f"[strata] Ip edges {qi} asym edges {qa}")


if __name__ == "__main__":
    main()
