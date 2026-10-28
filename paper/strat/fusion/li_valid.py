"""Doc 35 §37o: score the li warp (on top of the §37n RZ-head shift) on the 180-shot VALID sample (CPU).

    .venv/bin/python paper/strat/fusion/li_valid.py   ->  runs/strat/fusion/parts_LI-<arm>.pkl, li_valid.json
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import strategies as ST
import axis_shift as AX
import li_warp as LW

ARMS = {"li-avg": "avg", "li-head": "head"}


def main() -> None:
    sota, d = fx.data()
    sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    from common import AXIS_SIGN
    shots, grp = ST.sample()
    parts_e, nm = ST.load_entry()
    parts_rz = pickle.load(open(fx.OUT / "parts_AX-RZ-head.pkl", "rb"))
    f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib, ir_, iz_, il_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis", "efit_li"))
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    shifter = AX.make_shifter(R, Z, mask, sign); warp = LW.make_li_warp(R, Z, mask, sign)
    rep = {}
    for arm, mode in ARMS.items():
        pf = fx.OUT / f"parts_LI-{arm}.pkl"; betas = []
        if pf.exists():
            parts = pickle.load(open(pf, "rb"))
        else:
            def pred(s):
                cs, ss = f(s); sc = ss.mean(0)
                psi = (cs.mean(0) @ comps + mean).reshape(-1, 65, 65)
                out = np.empty_like(psi)
                for k in range(len(psi)):
                    a, _ = shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")
                    out[k], beta, _ = warp(a, sc[k, il_], full=(mode == "head"))
                    betas.append(beta)
                return out.reshape(len(psi), -1), sc[:, iq], sc[:, ib]
            parts = fx.frame_parts(d, shots, pred, np.asarray(nm["c_mean"]))
            pickle.dump(parts, open(pf, "wb"))
            b = np.abs(np.array(betas))
            print(f"[s37o] {arm}: |beta| median {np.median(b):.4f} at cap {np.mean(b >= LW.BETA_CAP - 1e-9):.4f}", flush=True)
        for ref_name, ref in (("RZ-head", parts_rz), ("entry", parts_e)):
            r = ST.compare(ref, parts, shots, grp, nm, d); rep[f"{arm} vs {ref_name}"] = r; o = r["overall"]
            print(f"[s37o] {arm} vs {ref_name}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
                  f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})} cons {({k: round(v, 4) for k, v in o['d_cons_each'].items()})}", flush=True)
    (fx.OUT / "li_valid.json").write_text(json.dumps(rep, indent=1, default=float))
    best = max(ARMS, key=lambda a: rep[f"{a} vs RZ-head"]["overall"]["S_new"]); o = rep[f"{best} vs RZ-head"]["overall"]
    print(f"S37O_VALID best={best} delta={o['delta']:+.5f} lo={o['lo']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
