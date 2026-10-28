"""Doc 35 §37c: score the matched retrains (retrain_fix.py) on the registered 180-shot stratified VALID sample (CPU).

    .venv/bin/python paper/strat/fusion/score_fix.py   ->  runs/strat/fusion/parts_FIX-<arm>-s<k>.pkl, fix_report.json

Pairs: ip vs old and iplag vs old per seed (the registered comparisons), plus old vs the stored entry-member trunk of
the same seed (sanity: retraining with the current code should land near it). Per-frame official-S parts and the
paired shot bootstrap are strategies.py's own.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

import fx  # noqa: I001  (sets CUDA_VISIBLE_DEVICES="" first)
import strategies as ST
from retrain_fix import data_for


def parts_for(path, dd, d, shots, nm, comps, mean, iq, ib):
    mp_ = fx.member_predict(path, dd, {s: fx.valid_rows(d, s) for s in shots})

    def pred(s):
        c, sc = mp_[s]
        return c.astype(np.float64) @ comps + mean, sc[:, iq].astype(np.float64), sc[:, ib].astype(np.float64)
    return fx.frame_parts(d, shots, pred, np.asarray(nm["c_mean"]))


def main() -> None:
    sota, d = fx.data()
    shots, grp = ST.sample()
    parts_e, nm = ST.load_entry()
    _, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    parts = {}
    for f in sorted(fx.OUT.glob("fix_*_s*.pt")):
        arm, seed = f.stem.split("_")[1], int(f.stem.split("_s")[-1])
        name = f"FIX-{arm}-s{seed}"; pf = fx.OUT / f"parts_{name}.pkl"
        if pf.exists():
            parts[name] = pickle.load(open(pf, "rb")); continue
        parts[name] = parts_for(f, d if arm == "old" else data_for(arm), d, shots, nm, comps, mean, iq, ib)
        pickle.dump(parts[name], open(pf, "wb"))
        S = fx.composite(np.stack([fx.shot_sums(parts[name][s]) for s in shots]), nm)["S"]
        print(f"[score-fix] {name}: sample S {S:.5f}", flush=True)
    for seed in (1, 2):
        pf = fx.OUT / f"parts_OLD-s{seed}.pkl"
        if not pf.exists() and (fx.TRUNKS / f"trunk1_t5-ew2_seed{seed}.pt").exists():
            pickle.dump(parts_for(fx.TRUNKS / f"trunk1_t5-ew2_seed{seed}.pt", d, d, shots, nm, comps, mean, iq, ib), open(pf, "wb"))
        if pf.exists():
            parts[f"OLD-s{seed}"] = pickle.load(open(pf, "rb"))
    rep = {}
    for seed in (1, 2):
        for new, ref in ((f"FIX-ip-s{seed}", f"FIX-old-s{seed}"), (f"FIX-iplag-s{seed}", f"FIX-old-s{seed}"),
                         (f"FIX-iplag-s{seed}", f"FIX-ip-s{seed}"), (f"FIX-old-s{seed}", f"OLD-s{seed}"),
                         (f"FIX-oldk-s{seed}", f"FIX-old-s{seed}"), (f"FIX-iplagk-s{seed}", f"FIX-old-s{seed}"),
                         (f"FIX-iplagk50-s{seed}", f"FIX-old-s{seed}"), (f"FIX-iplagk-s{seed}", f"FIX-iplag-s{seed}"),
                         (f"FIX-oldop01-s{seed}", f"FIX-old-s{seed}"), (f"FIX-oldop1-s{seed}", f"FIX-old-s{seed}")):
            if new in parts and ref in parts:
                r = ST.compare(parts[ref], parts[new], shots, grp, nm, d)
                rep[f"{new} vs {ref}"] = r; o = r["overall"]
                print(f"[score-fix] {new} vs {ref}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} "
                      f"[{o['lo']:+.5f}, {o['hi']:+.5f}] terms {({k: round(v, 5) for k, v in o['d_terms'].items()})}", flush=True)
                for k, v in r["phases"].items():
                    print(f"    phase {k:10s} ref {v['S_ref']:.5f} delta {v['delta']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]", flush=True)
    (fx.OUT / "fix_report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
