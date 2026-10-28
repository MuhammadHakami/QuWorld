"""Doc 35 §38d: psi-specialist ensemble (3 PSI-A seeds) with the entry's q95/betaN, on the 180-shot VALID sample:
(a) no post-processing vs the entry; (b) + the frozen post-processing (axis shift on the entry's axis heads, boundary
warp to the 6-seed v8 head radii, elongation, li warp) vs the frozen entry pipeline (parts_BW-Q-bkl3_v8_6s.pkl). CPU.

    .venv/bin/python paper/strat/fusion/s38d_eval.py   ->  runs/strat/fusion/s38d_eval.json
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import strategies as ST
import axis_shift as AX
import boundary_warp as BW
import kappa_warp as KW
import li_warp as LW

SOTA = fx.ROOT / "runs" / "sophelio_sota"
PSI = ["s38c_PSI-A.pt", "s38c_PSI-A-s1.pt", "s38c_PSI-A-s3.pt"]


def main() -> None:
    sota, d = fx.data()
    sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    from common import AXIS_SIGN
    shots, grp = ST.sample(); parts_e, nm = ST.load_entry()
    f, comps64, mean, H = ST.entry_preds(d, sota)
    rows = {s: fx.valid_rows(d, s) for s in shots}
    comps_all = d.meta["comps"].numpy().astype(np.float64)
    mods = [fx.member_predict(SOTA / p, d, rows) for p in PSI]
    iq, ib, ir_, iz_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis"))
    lab = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    Ph = np.mean([np.load(fx.OUT / f"head_boundary_v8_s{s}.npz")["pred_valid"] for s in range(1, 7)], 0).astype(np.float64)
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    sh = AX.make_shifter(R, Z, mask, sign); bw = BW.make_boundary_warp(R, Z, mask, sign)
    kw = KW.make_kappa_warp(R, Z, mask, sign); liw = LW.make_li_warp(R, Z, mask, sign)

    def make(post):
        def pred(s):
            c = np.mean([m[s][0] for m in mods], 0).astype(np.float64)
            psi = (c @ comps_all[:c.shape[1]] + mean).reshape(-1, 65, 65)
            sc = f(s)[1].mean(0)                                           # entry ensemble scalars (q95, betaN, axis)
            if post:
                out = np.empty_like(psi)
                for k, r in enumerate(rows[s]):
                    i = pos[int(r)]
                    a, _ = sh(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")
                    a, _ = bw(a, Ph[i, 7:]); a, _ = kw(a, Ph[i, 2])
                    out[k] = liw(a, Ph[i, 6], full=True)[0]
                psi = out
            return psi.reshape(len(psi), -1), sc[:, iq].astype(np.float64), sc[:, ib].astype(np.float64)
        return pred
    rep = {}
    for name, post, ref_name in (("C1", False, "entry"), ("C1post", True, "frozen")):
        pf = fx.OUT / f"parts_S38D-{name}.pkl"
        parts = pickle.load(open(pf, "rb")) if pf.exists() else fx.frame_parts(d, shots, make(post), np.asarray(nm["c_mean"]))
        pickle.dump(parts, open(pf, "wb"))
        ref = parts_e if ref_name == "entry" else pickle.load(open(fx.OUT / "parts_BW-Q-bkl3_v8_6s.pkl", "rb"))
        r = ST.compare(ref, parts, shots, grp, nm, d); o = r["overall"]; rep[name] = r
        print(f"[s38d] {name} vs {ref_name}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})}", flush=True)
    (fx.OUT / "s38d_eval.json").write_text(json.dumps(rep, indent=1, default=float))
    print("S38D_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
