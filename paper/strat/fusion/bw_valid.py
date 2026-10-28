"""Doc 35 §37q: score one boundary-warp arm on the 180-shot VALID sample vs the §37o pipeline (CPU).

    .venv/bin/python paper/strat/fusion/bw_valid.py <Q-b|Q-bl|Q-bl3>   ->  parts_BW-<arm>.pkl, bw_valid_<arm>.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import strategies as ST
import axis_shift as AX
import li_warp as LW
import boundary_warp as BW
import kappa_warp as KW


def main() -> None:
    arm = sys.argv[1]; assert arm in ("Q-b", "Q-bl", "Q-bl3", "Q-bkl3")
    sota, d = fx.data()
    sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    from common import AXIS_SIGN
    shots, grp = ST.sample(); parts_e, nm = ST.load_entry()
    tag = os.environ.get("STRAT_LABEL_SUFFIX", "")                            # §37s: "_v2" heads (64 rays, all TRAIN)
    ref_file, ref_label = ("parts_BW-Q-bl3.pkl", "Q-bl3") if tag else ("parts_LI-li-avg.pkl", "li-avg")
    if arm == "Q-bkl3":                                                         # §37w is compared with Q-bl3 on the same heads
        ref_file, ref_label = f"parts_BW-Q-bl3{tag}.pkl", f"Q-bl3{tag}"
    if os.environ.get("STRAT_HEAD_SEEDS", "1,2,3") != "1,2,3":                  # §37y: same arm with the 3-seed heads
        ref_file, ref_label = f"parts_BW-{arm}{tag}.pkl", f"{arm}{tag}"
    if os.environ.get("STRAT_REF"):                                             # §37ab: explicit reference parts (e.g. Q-bkl3_v5)
        ref_label = os.environ["STRAT_REF"]; ref_file = f"parts_BW-{ref_label}.pkl"
    ref = pickle.load(open(fx.OUT / ref_file, "rb"))
    f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib, ir_, iz_, il_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis", "efit_li"))
    lab = np.load(fx.OUT / f"boundary_labels{tag}.npz"); pos = {int(r): i for i, r in enumerate(lab["rows_valid"])}
    seeds = [int(x) for x in os.environ.get("STRAT_HEAD_SEEDS", "1,2,3").split(",")]   # §37y: 1-6
    Ph = np.mean([np.load(fx.OUT / f"head_boundary{tag}_s{s}.npz")["pred_valid"] for s in seeds], 0).astype(np.float64)
    stag = "" if seeds == [1, 2, 3] else f"_{len(seeds)}s"
    R, Z, mask = AX.grid(); sign = AXIS_SIGN["DIII-D"]
    shifter = AX.make_shifter(R, Z, mask, sign); liw = LW.make_li_warp(R, Z, mask, sign); bw = BW.make_boundary_warp(R, Z, mask, sign)
    kw = KW.make_kappa_warp(R, Z, mask, sign)
    moved = []

    def pred(s):
        cs, ss = f(s); sc = ss.mean(0); rows = fx.valid_rows(d, s)
        psi = (cs.mean(0) @ comps + mean).reshape(-1, 65, 65); out = np.empty_like(psi)
        for k in range(len(psi)):
            i = pos[int(rows[k])]
            a, _ = shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), "RZ", "head")
            a, rms = bw(a, Ph[i, 7:]); moved.append(rms)
            if arm == "Q-bkl3":                                                 # §37w: elongation to the kappa head, then li
                a, _ = kw(a, Ph[i, 2])
                a, _, _ = liw(a, Ph[i, 6], full=True)
            elif arm == "Q-bl":
                a, _, _ = liw(a, sc[k, il_])
            elif arm == "Q-bl3":
                a, _, _ = liw(a, Ph[i, 6], full=True)
            out[k] = a
        return out.reshape(len(psi), -1), sc[:, iq], sc[:, ib]
    pf = fx.OUT / f"parts_BW-{arm}{tag}{stag}.pkl"
    parts = fx.frame_parts(d, shots, pred, np.asarray(nm["c_mean"]))
    pickle.dump(parts, open(pf, "wb"))
    mv = np.array(moved); print(f"[s37q] {arm}: warp rms |r_t - r_p| median {np.nanmedian(mv) * 1000:.1f} mm, skipped {np.isnan(mv).mean():.4f}", flush=True)
    rep = {}
    for ref_name, rp in ((ref_label, ref), ("entry", parts_e)):
        r = ST.compare(rp, parts, shots, grp, nm, d); rep[ref_name] = r; o = r["overall"]
        print(f"[s37q] {arm} vs {ref_name}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})} cons {({k: round(v, 4) for k, v in o['d_cons_each'].items()})}", flush=True)
        for ph, v in r["phases"].items():
            print(f"    phase {ph:10s} delta {v['delta']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]", flush=True)
    (fx.OUT / f"bw_valid_{arm}{tag}{stag}.json").write_text(json.dumps(rep, indent=1, default=float))
    o = rep[ref_label]["overall"]
    print(f"{'S37S_VALID' if tag else 'S37Q_VALID'} arm={arm}{tag}{stag} S={o['S_new']:.5f} delta={o['delta']:+.5f} lo={o['lo']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
