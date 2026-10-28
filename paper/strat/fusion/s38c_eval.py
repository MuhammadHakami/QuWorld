"""Doc 35 §38c: specialist pair (psi from PSI-A, q95/betaN from QB-A) vs the best single entry member and vs the 7-member
entry, on the 180-shot VALID sample with the official per-frame parts (no post-processing). CPU.

    .venv/bin/python paper/strat/fusion/s38c_eval.py   ->  runs/strat/fusion/s38c_eval.json
"""
from __future__ import annotations

import json
import pickle

import numpy as np

import fx  # noqa: I001  (CPU only)
import strategies as ST

SOTA = fx.ROOT / "runs" / "sophelio_sota"


def main() -> None:
    sota, d = fx.data()
    shots, grp = ST.sample(); parts_e, nm = ST.load_entry()
    rows = {s: fx.valid_rows(d, s) for s in shots}
    H = sota.HEADS; iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    mean = d.meta["mean"].numpy().astype(np.float64)
    comps_all = d.meta["comps"].numpy().astype(np.float64)
    P = fx.member_predict(SOTA / "s38c_PSI-A.pt", d, rows)
    Q = fx.member_predict(SOTA / "s38c_QB-A.pt", d, rows)
    M = fx.member_predict(fx.TRUNKS / "trunk1_t19-ew2_seed2.pt", d, rows)

    def pred_from(Pp, Qq):
        def pred(s):
            c, _ = Pp[s]; _, sc = Qq[s]
            return c.astype(np.float64) @ comps_all[:c.shape[1]] + mean, sc[:, iq].astype(np.float64), sc[:, ib].astype(np.float64)
        return pred
    rep = {}
    for name, pf in (("pair", pred_from(P, Q)), ("member", pred_from(M, M)), ("psiA_member_qb", pred_from(P, M))):
        f = fx.OUT / f"parts_S38C-{name}.pkl"
        parts = pickle.load(open(f, "rb")) if f.exists() else fx.frame_parts(d, shots, pf, np.asarray(nm["c_mean"]))
        pickle.dump(parts, open(f, "wb")); rep[name] = parts
    out = {}
    for a, b in (("member", "pair"), ("entry", "pair"), ("member", "psiA_member_qb")):
        r = ST.compare(parts_e if a == "entry" else rep[a], rep[b], shots, grp, nm, d); o = r["overall"]; out[f"{b} vs {a}"] = r
        print(f"[s38c] {b} vs {a}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})} r2q95 {o['d_r2_q95']:+.4f} r2bN {o['d_r2_betaN']:+.4f}", flush=True)
    for name, parts in rep.items():
        c = fx.composite(np.stack([fx.shot_sums(parts[s]) for s in shots]), nm)
        print(f"[s38c] {name}: S {c['S']:.5f} r2_psi {c['r2_psi']:.6f} qb {c['r2_qb']:.5f} dlcfs {c['dlcfs']:.5f} cons {c['consistency']:.5f}", flush=True)
    (fx.OUT / "s38c_eval.json").write_text(json.dumps(out, indent=1, default=float))
    print("S38C_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
