"""Doc 35 §37n/o follow-up: what does float16 storage of psi (the leaderboard file format) cost on the 180-shot VALID
sample? Entry and entry + RZ-head shift + li-avg warp, each in float32 vs float16 (the scorer upcasts on read).

    .venv/bin/python paper/strat/fusion/fp16_check.py   ->  runs/strat/fusion/fp16_check.json
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

import fx  # noqa: I001
import strategies as ST
import axis_shift as AX
import li_warp as LW


def main() -> None:
    sota, d = fx.data()
    sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    from common import AXIS_SIGN
    shots, grp = ST.sample(); parts_e, nm = ST.load_entry()
    f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib, ir_, iz_, il_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis", "efit_li"))
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    shifter = AX.make_shifter(R, Z, mask, sign); warp = LW.make_li_warp(R, Z, mask, sign)
    rep = {}
    for arm in ("entry16", "corr16"):
        pf = fx.OUT / f"parts_FP-{arm}.pkl"
        if pf.exists():
            parts = pickle.load(open(pf, "rb"))
        else:
            def pred(s):
                cs, ss = f(s); sc = ss.mean(0)
                psi = (cs.mean(0) @ comps + mean).reshape(-1, 65, 65)
                if arm == "corr16":
                    psi = np.stack([warp(shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")[0], sc[k, il_])[0]
                                    for k in range(len(psi))])
                return psi.astype(np.float32).astype(np.float16).astype(np.float64).reshape(len(psi), -1), sc[:, iq], sc[:, ib]
            parts = fx.frame_parts(d, shots, pred, np.asarray(nm["c_mean"]))
            pickle.dump(parts, open(pf, "wb"))
        ref = parts_e if arm == "entry16" else pickle.load(open(fx.OUT / "parts_LI-li-avg.pkl", "rb"))
        r = ST.compare(ref, parts, shots, grp, nm, d); rep[arm] = r; o = r["overall"]
        print(f"[fp16] {arm} vs float64: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})} cons {({k: round(v, 4) for k, v in o['d_cons_each'].items()})}", flush=True)
    (fx.OUT / "fp16_check.json").write_text(json.dumps(rep, indent=1, default=float))
    print("FP16_DONE", flush=True)


if __name__ == "__main__":
    main()
