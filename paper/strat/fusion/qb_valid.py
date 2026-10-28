"""Doc 35 §37r: dedicated q95 / betaN heads vs the entry's trunk heads on the 180-shot VALID sample (CPU, no psi change).

    .venv/bin/python paper/strat/fusion/qb_valid.py   ->  runs/strat/fusion/qb_valid.json
"""
from __future__ import annotations

import json

import numpy as np

import fx  # noqa: I001
import strategies as ST


def main() -> None:
    sota, d = fx.data()
    shots, grp = ST.sample(); parts_e, nm = ST.load_entry()
    f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    hs = [np.load(fx.OUT / f"head_qb_s{s}.npz") for s in (1, 2, 3)]
    va = hs[0]["rows_valid"]; pos = {int(r): i for i, r in enumerate(va)}
    ph = np.mean([h["pred_valid"] for h in hs], 0).astype(np.float64)
    rep = {}
    for arm in ("H-qb", "H-avg"):
        out = {}
        for s in shots:
            rows = fx.valid_rows(d, s); ii = [pos[int(r)] for r in rows]
            sc = f(s)[1].mean(0); q, b = ph[ii, 0], ph[ii, 1]
            if arm == "H-avg":
                q, b = 0.5 * (q + sc[:, iq]), 0.5 * (b + sc[:, ib])
            p = dict(parts_e[s]); p["q_res"] = (p["q_gt"] - q) ** 2; p["b_res"] = (p["b_gt"] - b) ** 2
            out[s] = p
        r = ST.compare(parts_e, out, shots, grp, nm, d); rep[arm] = r; o = r["overall"]
        print(f"[s37r] {arm}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"r2_q95 {o['d_r2_q95']:+.5f} r2_betaN {o['d_r2_betaN']:+.5f}", flush=True)
    (fx.OUT / "qb_valid.json").write_text(json.dumps(rep, indent=1, default=float))
    best = max(rep, key=lambda a: rep[a]["overall"]["S_new"]); o = rep[best]["overall"]
    print(f"S37R_VALID best={best} delta={o['delta']:+.5f} lo={o['lo']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
